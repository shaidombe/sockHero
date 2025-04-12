export interface Color {
    r: number;
    g: number;
    b: number;
}

export interface Region {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    color: Color;
    pixels: Set<string>;
    size: number;
    width: number;
    height: number;
    texture: Texture;
}

export interface Settings {
    gridSize: number;
    minRegionSize: number;
    maxRegionSize: number;
    colorThreshold: number;
    sizeRatioThreshold: number;
    aspectRatioThreshold: number;
    textureThreshold: number;
}

interface Texture {
    contrast: number;
    pattern: number[];
    edgeCount: number;
    avgBrightness: number;
    verticalEdges: number;
    horizontalEdges: number;
    patternDirection: 'vertical' | 'horizontal' | 'both' | 'none';
}

function getPixelColor(imageData: ImageData, x: number, y: number): Color {
    const i = (y * imageData.width + x) * 4;
    return {
        r: imageData.data[i],
        g: imageData.data[i + 1],
        b: imageData.data[i + 2]
    };
}

function colorDifference(color1: Color, color2: Color): number {
    return Math.abs(color1.r - color2.r) +
           Math.abs(color1.g - color2.g) +
           Math.abs(color1.b - color2.b);
}

function isSurfaceColor(color: Color, surfaceColors: Color[], threshold: number): boolean {
    return surfaceColors.some(surfaceColor => 
        colorDifference(color, surfaceColor) < threshold
    );
}

function getTextureFeatures(imageData: ImageData, region: Region): Texture {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    
    // Sample multiple areas of the sock
    const areas = [
        // Top area
        {
            x: Math.floor((region.minX + region.maxX) / 2),
            y: Math.floor(region.minY + region.height * 0.25),
            size: Math.min(region.width, region.height) / 4
        },
        // Middle area
        {
            x: Math.floor((region.minX + region.maxX) / 2),
            y: Math.floor((region.minY + region.maxY) / 2),
            size: Math.min(region.width, region.height) / 4
        },
        // Bottom area
        {
            x: Math.floor((region.minX + region.maxX) / 2),
            y: Math.floor(region.maxY - region.height * 0.25),
            size: Math.min(region.width, region.height) / 4
        }
    ];

    let totalPattern: number[] = [];
    let totalContrast = 0;
    let totalEdgeCount = 0;
    let totalBrightness = 0;
    let totalSamples = 0;
    let verticalEdges = 0;
    let horizontalEdges = 0;

    for (const area of areas) {
        const sampleSize = Math.min(area.size, Math.min(width, height) / 4);
        const step = Math.max(1, Math.floor(sampleSize / 25)); // Even smaller step for more detail

        // Sample in a grid pattern
        for (let i = -sampleSize; i <= sampleSize; i += step) {
            for (let j = -sampleSize; j <= sampleSize; j += step) {
                const px = Math.max(0, Math.min(width - 1, area.x + i));
                const py = Math.max(0, Math.min(height - 1, area.y + j));
                const idx = (py * width + px) * 4;
                
                const gray = (data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114);
                totalPattern.push(gray);
                totalBrightness += gray;
                totalSamples++;

                // Enhanced edge detection with direction
                if (i > -sampleSize && j > -sampleSize) {
                    // Check vertical edges
                    const verticalPx = Math.max(0, Math.min(width - 1, area.x + i));
                    const verticalPy = Math.max(0, Math.min(height - 1, area.y + j - step));
                    const verticalIdx = (verticalPy * width + verticalPx) * 4;
                    const verticalGray = (data[verticalIdx] * 0.299 + data[verticalIdx + 1] * 0.587 + data[verticalIdx + 2] * 0.114);
                    const verticalDiff = Math.abs(gray - verticalGray);
                    
                    // Check horizontal edges
                    const horizontalPx = Math.max(0, Math.min(width - 1, area.x + i - step));
                    const horizontalPy = Math.max(0, Math.min(height - 1, area.y + j));
                    const horizontalIdx = (horizontalPy * width + horizontalPx) * 4;
                    const horizontalGray = (data[horizontalIdx] * 0.299 + data[horizontalIdx + 1] * 0.587 + data[horizontalIdx + 2] * 0.114);
                    const horizontalDiff = Math.abs(gray - horizontalGray);

                    totalContrast += Math.max(verticalDiff, horizontalDiff);
                    
                    if (verticalDiff > 10) {
                        verticalEdges++;
                        totalEdgeCount++;
                    }
                    if (horizontalDiff > 10) {
                        horizontalEdges++;
                        totalEdgeCount++;
                    }
                }
            }
        }
    }

    // Determine pattern direction
    const verticalRatio = verticalEdges / totalSamples;
    const horizontalRatio = horizontalEdges / totalSamples;
    let patternDirection: 'vertical' | 'horizontal' | 'both' | 'none' = 'none';
    
    if (verticalRatio > 0.1 && horizontalRatio > 0.1) {
        patternDirection = 'both';
    } else if (verticalRatio > 0.1) {
        patternDirection = 'vertical';
    } else if (horizontalRatio > 0.1) {
        patternDirection = 'horizontal';
    }

    return {
        contrast: totalContrast / totalSamples,
        pattern: totalPattern,
        edgeCount: totalEdgeCount,
        avgBrightness: totalBrightness / totalSamples,
        verticalEdges,
        horizontalEdges,
        patternDirection
    };
}

function textureDifference(t1: Texture, t2: Texture): number {
    // Compare patterns using normalized cross-correlation
    let maxCorrelation = 0;
    const windowSize = Math.floor(Math.min(t1.pattern.length, t2.pattern.length) * 0.7);
    
    // Normalize patterns
    const normalize = (pattern: number[]) => {
        const mean = pattern.reduce((a, b) => a + b, 0) / pattern.length;
        const std = Math.sqrt(pattern.reduce((a, b) => a + (b - mean) ** 2, 0) / pattern.length);
        return pattern.map(x => (x - mean) / (std || 1));
    };

    const norm1 = normalize(t1.pattern);
    const norm2 = normalize(t2.pattern);

    // Calculate correlation with sliding window
    for (let offset = 0; offset < t1.pattern.length - windowSize; offset++) {
        let correlation = 0;
        for (let i = 0; i < windowSize; i++) {
            correlation += norm1[offset + i] * norm2[i];
        }
        correlation /= windowSize;
        maxCorrelation = Math.max(maxCorrelation, Math.abs(correlation));
    }

    // Compare edge density and direction
    const edgeDensity1 = t1.edgeCount / t1.pattern.length;
    const edgeDensity2 = t2.edgeCount / t2.pattern.length;
    const edgeRatio = Math.min(edgeDensity1, edgeDensity2) / Math.max(edgeDensity1, edgeDensity2);
    
    // Compare pattern direction
    const directionScore = t1.patternDirection === t2.patternDirection ? 1 : 
        (t1.patternDirection === 'both' || t2.patternDirection === 'both' ? 0.7 : 0.3);
    
    // Compare local contrast patterns
    const contrastRatio = Math.min(t1.contrast, t2.contrast) / Math.max(t1.contrast, t2.contrast);
    
    // Weighted combination focusing on pattern correlation and direction
    return (
        maxCorrelation * 0.4 +     // Pattern correlation
        edgeRatio * 0.2 +          // Edge density similarity
        directionScore * 0.3 +     // Pattern direction similarity
        contrastRatio * 0.1        // Local contrast similarity
    );
}

export function findColorRegions(imageData: ImageData, settings: Settings): Region[] {
    const regions: Region[] = [];
    const visited = new Set<string>();
    const width = imageData.width;
    const height = imageData.height;

    // Scan the image with larger steps to find potential sock regions
    const scanStep = Math.max(20, settings.gridSize);
    
    for (let y = 0; y < height; y += scanStep) {
        for (let x = 0; x < width; x += scanStep) {
            if (visited.has(`${x},${y}`)) continue;

            const color = getPixelColor(imageData, x, y);
            const brightness = (color.r + color.g + color.b) / 3;
            
            // Skip white or very dark areas
            if (brightness > 240 || brightness < 15) continue;

            // Try to find a complete sock region
            const region = growRegion(imageData, x, y, color, visited, settings);
            if (region) {
                const aspectRatio = region.height / region.width;
                // A sock typically has height about 2-4 times its width
                if (aspectRatio >= 1.5 && aspectRatio <= 4 && 
                    region.height >= 100 && // Minimum height for a sock
                    region.width >= 50) {   // Minimum width for a sock
                    regions.push(region);
                }
            }
        }
    }

    console.log(`Found ${regions.length} potential socks`);
    return regions;
}

function growRegion(
    imageData: ImageData,
    startX: number,
    startY: number,
    startColor: Color,
    visited: Set<string>,
    settings: Settings
): Region | null {
    const region: Region = {
        minX: startX,
        maxX: startX,
        minY: startY,
        maxY: startY,
        color: startColor,
        pixels: new Set([`${startX},${startY}`]),
        size: 1,
        width: 1,
        height: 1,
        texture: { contrast: 0, pattern: [], edgeCount: 0, avgBrightness: 0, verticalEdges: 0, horizontalEdges: 0, patternDirection: 'none' }
    };

    const queue: [number, number][] = [[startX, startY]];
    const width = imageData.width;
    const height = imageData.height;

    while (queue.length > 0 && region.size < settings.maxRegionSize) {
        const [x, y] = queue.shift()!;
        
        // Check 8 neighboring pixels
        const neighbors = [
            [x+1, y], [x-1, y], [x, y+1], [x, y-1],
            [x+1, y+1], [x-1, y-1], [x+1, y-1], [x-1, y+1]
        ];

        for (const [nx, ny] of neighbors) {
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            
            const key = `${nx},${ny}`;
            if (visited.has(key)) continue;
            
            const neighborColor = getPixelColor(imageData, nx, ny);
            const colorDiff = colorDifference(neighborColor, startColor);
            
            // More lenient color threshold for region growing
            if (colorDiff <= settings.colorThreshold * 1.8) {
        visited.add(key);
        region.pixels.add(key);
                queue.push([nx, ny]);

        // Update region bounds
                region.minX = Math.min(region.minX, nx);
                region.maxX = Math.max(region.maxX, nx);
                region.minY = Math.min(region.minY, ny);
                region.maxY = Math.max(region.maxY, ny);
                region.size++;
            }
        }
    }

    // Update dimensions
    region.width = region.maxX - region.minX + 1;
    region.height = region.maxY - region.minY + 1;

    // Calculate average color
    let totalR = 0, totalG = 0, totalB = 0;
    region.pixels.forEach(key => {
        const [x, y] = key.split(',').map(Number);
        const color = getPixelColor(imageData, x, y);
        totalR += color.r;
        totalG += color.g;
        totalB += color.b;
    });

    region.color = {
        r: Math.round(totalR / region.size),
        g: Math.round(totalG / region.size),
        b: Math.round(totalB / region.size)
    };

    // Return null if region is too small
    if (region.size < settings.minRegionSize * 0.7) { // More lenient minimum size
        return null;
    }

    return region;
}

export function findMatchingPairs(regions: Region[], settings: Settings, imageData: ImageData): [Region, Region][] {
    type Match = {
        r1: Region;
        r2: Region;
        score: number;
        colorScore: number;
        textureScore: number;
        sizeScore: number;
    };
    const allMatches: Match[] = [];
    
    const sortedRegions = [...regions].sort((a, b) => b.size - a.size);

    for (let i = 0; i < sortedRegions.length; i++) {
        const r1 = sortedRegions[i];
        const texture1 = getTextureFeatures(imageData, r1);

        for (let j = i + 1; j < sortedRegions.length; j++) {
            const r2 = sortedRegions[j];

            // Skip if regions overlap
            const overlap = Math.max(0,
                Math.min(r1.maxX, r2.maxX) - Math.max(r1.minX, r2.minX)
            ) * Math.max(0,
                Math.min(r1.maxY, r2.maxY) - Math.max(r1.minY, r2.minY)
            );
            const minArea = Math.min(r1.size, r2.size);
            if (overlap / minArea > 0.3) continue;

            const texture2 = getTextureFeatures(imageData, r2);

            // Compare sizes
            const heightRatio = Math.min(r1.height, r2.height) / Math.max(r1.height, r2.height);
            const widthRatio = Math.min(r1.width, r2.width) / Math.max(r1.width, r2.width);
            const sizeRatio = Math.min(r1.size, r2.size) / Math.max(r1.size, r2.size);

            if (heightRatio < 0.5 || widthRatio < 0.5 || sizeRatio < 0.5) continue;
            const sizeScore = (heightRatio + widthRatio + sizeRatio) / 3;

            // Check if both socks are dark (black/navy)
            const isDark1 = (r1.color.r + r1.color.g + r1.color.b) / 3 < 80;
            const isDark2 = (r2.color.r + r2.color.g + r2.color.b) / 3 < 80;

            // Calculate color score differently for dark socks
            let colorScore;
            if (isDark1 && isDark2) {
                // For dark socks, focus more on brightness similarity
                const brightness1 = (r1.color.r + r1.color.g + r1.color.b) / 3;
                const brightness2 = (r2.color.r + r2.color.g + r2.color.b) / 3;
                colorScore = 1 - Math.abs(brightness1 - brightness2) / 80;
            } else {
                const colorDiff = colorDifference(r1.color, r2.color);
                colorScore = Math.max(0, 1 - (colorDiff / (settings.colorThreshold * 2)));
            }

            const textureScore = textureDifference(texture1, texture2);
            
            // Calculate final score with emphasis on texture for patterned socks
            const hasStrongPattern = texture1.edgeCount > 100 || texture2.edgeCount > 100;
            const score = hasStrongPattern ? (
                textureScore * 0.7 +    // Much higher weight on texture for patterned socks
                colorScore * 0.1 +      // Less weight on color
                sizeScore * 0.2         // Keep size importance
            ) : isDark1 && isDark2 ? (
                textureScore * 0.6 +    // Higher weight on texture for dark socks
                colorScore * 0.2 +      // Less weight on color
                sizeScore * 0.2         // Keep size importance
            ) : (
                textureScore * 0.4 +    // Normal weight for plain colored socks
                colorScore * 0.4 +
                sizeScore * 0.2
            );

            // Adjust thresholds based on pattern strength
            const minScore = hasStrongPattern ? 0.4 : (isDark1 && isDark2 ? 0.45 : 0.5);
            const minTextureScore = hasStrongPattern ? 0.3 : (isDark1 && isDark2 ? 0.35 : 0.4);

            if (score > minScore && textureScore > minTextureScore) {
                allMatches.push({ 
                    r1, r2, score,
                    colorScore,
                    textureScore,
                    sizeScore
                });
            }
        }
    }

    // Sort matches by score
    allMatches.sort((a, b) => {
        // For very close scores, prefer matches with better texture scores
        if (Math.abs(b.score - a.score) < 0.1) {
            return b.textureScore - a.textureScore;
        }
        return b.score - a.score;
    });

    // Select best pairs
    const pairs: [Region, Region][] = [];
    const usedRegions = new Set<Region>();

    for (const match of allMatches) {
        if (!usedRegions.has(match.r1) && !usedRegions.has(match.r2)) {
            console.log('Found match:', {
                score: match.score.toFixed(2),
                colorScore: match.colorScore.toFixed(2),
                textureScore: match.textureScore.toFixed(2),
                sizeScore: match.sizeScore.toFixed(2),
                isDark: (match.r1.color.r + match.r1.color.g + match.r1.color.b) / 3 < 80,
                hasPattern: match.r1.texture?.edgeCount > 100 || match.r2.texture?.edgeCount > 100,
                pattern1: match.r1.texture?.patternDirection,
                pattern2: match.r2.texture?.patternDirection,
                r1: {
                    size: match.r1.size,
                    color: match.r1.color,
                    pos: `(${match.r1.minX},${match.r1.minY})`
                },
                r2: {
                    size: match.r2.size,
                    color: match.r2.color,
                    pos: `(${match.r2.minX},${match.r2.minY})`
                }
            });
            
            pairs.push([match.r1, match.r2]);
            usedRegions.add(match.r1);
            usedRegions.add(match.r2);
        }
    }

    return pairs;
} 