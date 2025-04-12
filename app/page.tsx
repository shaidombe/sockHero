'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import confetti from 'canvas-confetti';
import { findColorRegions, findMatchingPairs, type Settings, type Region, type Color } from './utils/sockDetection';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isFrozen, setIsFrozen] = useState(false);
  const [surfaceColors, setSurfaceColors] = useState<Color[]>([]);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [capturedFrame, setCapturedFrame] = useState<boolean>(false);
  const [settings, setSettings] = useState<Settings>({
    gridSize: 15,
    minRegionSize: 2000,
    maxRegionSize: 100000,
    colorThreshold: 35,
    sizeRatioThreshold: 1.2,
    aspectRatioThreshold: 0.2,
    textureThreshold: 30
  });
  const [isSelectingSurface, setIsSelectingSurface] = useState(false);
  const [touchPosition, setTouchPosition] = useState<{ x: number, y: number } | null>(null);
  const [selectedSurface, setSelectedSurface] = useState<{ x: number, y: number } | null>(null);
  const [canvasSize, setCanvasSize] = useState<{ width: number, height: number }>({ width: 1280, height: 720 });
  const [scanLine, setScanLine] = useState(0);
  const scanAnimationRef = useRef<number>(0);
  const [detectedRegions, setDetectedRegions] = useState<Region[]>([]);
  const [matchedPairs, setMatchedPairs] = useState<[Region, Region][]>([]);
  const frameProcessingRef = useRef<number>();
  const [isRealTimeDetection, setIsRealTimeDetection] = useState(false);
  const [isMainMenu, setIsMainMenu] = useState(true);
  const realTimeIntervalRef = useRef<NodeJS.Timeout>();
  const [lineWidth, setLineWidth] = useState(4);
  const [fontSize, setFontSize] = useState(24);
  const [isProcessing, setIsProcessing] = useState(false);
  const lastProcessingTime = useRef<number>(0);
  const processingTimeout = useRef<NodeJS.Timeout>();
  const [currentPairs, setCurrentPairs] = useState<[Region, Region][]>([]);

  const drawPairs = useCallback((ctx: CanvasRenderingContext2D, pairs: [Region, Region][]) => {
    const pairColors = [
      '#FF3366', '#33FF66', '#3366FF', '#FFCC33', '#FF33FF'
    ];

    pairs.forEach(([r1, r2]: [Region, Region], i) => {
      const pairColor = pairColors[i % pairColors.length];
      
      ctx.strokeStyle = pairColor;
      ctx.lineWidth = lineWidth;
      
      // Draw rectangles
      ctx.strokeRect(r1.minX, r1.minY, r1.maxX - r1.minX, r1.maxY - r1.minY);
      ctx.strokeRect(r2.minX, r2.minY, r2.maxX - r2.minX, r2.maxY - r2.minY);

      // Draw line
      const x1 = (r1.minX + r1.maxX) / 2;
      const y1 = (r1.minY + r1.maxY) / 2;
      const x2 = (r2.minX + r2.maxX) / 2;
      const y2 = (r2.minY + r2.maxY) / 2;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      // Draw pair number
      ctx.fillStyle = pairColor;
      ctx.font = `bold ${fontSize}px Arial`;
      ctx.fillText(`זוג ${i + 1}`, (x1 + x2) / 2, (y1 + y2) / 2);
    });
  }, [lineWidth, fontSize]);

  const processFrameWithDetection = useCallback(async () => {
    const now = Date.now();
    if (now - lastProcessingTime.current < 500) {
      return; // Skip if less than 500ms since last processing
    }

    if (!canvasRef.current || !videoRef.current || isProcessing) {
      return;
    }

    setIsProcessing(true);
    lastProcessingTime.current = now;

    const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    try {
      // Clear canvas and capture video frame
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      ctx.drawImage(videoRef.current, 0, 0);
      
      const imageData = ctx.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height);
      const regions = findColorRegions(imageData, settings);
      const pairs = findMatchingPairs(regions, settings, imageData);

      // Draw only the matched pairs
      const pairColors = [
        '#FF3366',  // ורוד-אדום
        '#33FF66',  // ירוק בהיר
        '#3366FF',  // כחול
        '#FFCC33',  // צהוב
        '#FF33FF'   // סגול
      ];

      pairs.forEach(([r1, r2]: [Region, Region], i) => {
        const pairColor = pairColors[i % pairColors.length];
        
        // Draw rectangles for the pair
        ctx.strokeStyle = pairColor;
        ctx.lineWidth = lineWidth;
        
        // Draw first sock rectangle
        ctx.strokeRect(
          r1.minX,
          r1.minY,
          r1.maxX - r1.minX,
          r1.maxY - r1.minY
        );
        
        // Draw second sock rectangle
        ctx.strokeRect(
          r2.minX,
          r2.minY,
          r2.maxX - r2.minX,
          r2.maxY - r2.minY
        );

        // Calculate centers for the line
        const x1 = (r1.minX + r1.maxX) / 2;
        const y1 = (r1.minY + r1.maxY) / 2;
        const x2 = (r2.minX + r2.maxX) / 2;
        const y2 = (r2.minY + r2.maxY) / 2;

        // Draw connecting line
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = pairColor;
        ctx.lineWidth = lineWidth;
        ctx.stroke();

        // Draw pair number
        ctx.fillStyle = pairColor;
        ctx.font = `bold ${fontSize}px Arial`;
        ctx.fillText(`זוג ${i + 1}`, (x1 + x2) / 2, (y1 + y2) / 2);
      });

      // Schedule next detection
      if (isRealTimeDetection) {
        setTimeout(processFrameWithDetection, 500);
      }

    } catch (error) {
      console.error('Error in frame processing:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [settings, lineWidth, fontSize, isRealTimeDetection]);

  // Effect for real-time detection
  useEffect(() => {
    let frameId: number;
    
    const updateCanvas = () => {
      if (!capturedFrame && canvasRef.current && videoRef.current) {
        const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
        if (ctx) {
          if (!isRealTimeDetection) {
            // If not in detection mode, just show the video
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            ctx.drawImage(videoRef.current, 0, 0);
          }
        }
      }
      frameId = requestAnimationFrame(updateCanvas);
    };

    updateCanvas();

    // Start detection if in real-time mode
    if (isRealTimeDetection) {
      processFrameWithDetection();
    }

    return () => {
      cancelAnimationFrame(frameId);
    };
  }, [capturedFrame, isRealTimeDetection, processFrameWithDetection]);

  const captureAndAnalyzeFrame = useCallback(() => {
    if (!canvasRef.current || !videoRef.current) {
      console.log('Missing refs:', { canvas: !!canvasRef.current, video: !!videoRef.current });
      return;
    }

    const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      console.log('No canvas context');
      return;
    }

    try {
      // Clear canvas and capture video frame
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      ctx.drawImage(videoRef.current, 0, 0);
      setCapturedFrame(true);

      console.log('Processing captured frame');
      
      // Get image data for the whole canvas
      const imageData = ctx.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // Find regions and pairs
      const regions = findColorRegions(imageData, settings);
      console.log('Found regions:', regions.length);
      
      const pairs = findMatchingPairs(regions, settings, imageData);
      console.log('Found pairs:', pairs.length);

      // Draw the pairs
      drawPairs(ctx, pairs);

    } catch (error) {
      console.error('Error in frame processing:', error);
    }
  }, [settings, drawPairs]);

  const resetCapture = useCallback(() => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    // Clear canvas and show live video
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    ctx.drawImage(videoRef.current, 0, 0);
    
    // Reset state
    setCapturedFrame(false);
    setDetectedRegions([]);
    setMatchedPairs([]);
  }, []);

  const startRealTimeDetection = () => {
    setIsMainMenu(false);
    setIsRealTimeDetection(true);
    setCapturedFrame(false);
  };

  const startManualDetection = () => {
    setIsMainMenu(false);
    setIsRealTimeDetection(false);
    setCapturedFrame(false);
  };

  const returnToMainMenu = () => {
    setIsMainMenu(true);
    setIsRealTimeDetection(false);
    setCapturedFrame(false);
    if (realTimeIntervalRef.current) {
      clearInterval(realTimeIntervalRef.current);
    }
  };

  // Update video frame continuously
  useEffect(() => {
    let animationFrame: number;

    const updateCanvas = () => {
      if (!capturedFrame && canvasRef.current && videoRef.current) {
        const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
        if (ctx) {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          ctx.drawImage(videoRef.current, 0, 0);
        }
        animationFrame = requestAnimationFrame(updateCanvas);
      }
    };

    if (!isRealTimeDetection) {
      updateCanvas();
    }

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [capturedFrame, isRealTimeDetection]);

  const initCamera = useCallback(async () => {
    try {
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      
      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: selectedCamera ? { exact: selectedCamera } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 },
          ...(isMobile ? { facingMode: { exact: 'environment' } } : {})
        }
      };

      console.log('Trying to access camera with constraints:', constraints);
      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        if (canvasRef.current) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
          
          const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
          if (ctx) {
            console.log('Canvas context created with willReadFrequently=true');
          }
        }
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('שגיאה בגישה למצלמה. אנא נסה שוב.');
    }
  }, [selectedCamera]);

  // Add effect to monitor selection mode changes
  useEffect(() => {
    console.log('Selection mode changed:', { isSelectingSurface });
  }, [isSelectingSurface]);

  const refreshCameras = async () => {
    try {
      setIsLoading(true);
      
      // Request permissions with advanced constraints
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 }
          // Remove facingMode constraint to support all cameras
        }
      };
      
      // First request access to trigger permissions
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      // Close the stream immediately
      stream.getTracks().forEach(track => track.stop());
      
      // Wait a bit for devices to be detected
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Then enumerate all devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      console.log('Found cameras:', videoDevices.map(d => ({ 
        label: d.label, 
        id: d.deviceId,
        groupId: d.groupId 
      })));
      
      setCameras(videoDevices);
      if (videoDevices.length > 0 && !selectedCamera) {
        // Try to find and select the back camera on mobile, otherwise use the first camera
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        if (isMobile) {
          const backCamera = videoDevices.find(device => 
            device.label.toLowerCase().includes('back') || 
            device.label.toLowerCase().includes('environment')
          );
          setSelectedCamera(backCamera?.deviceId || videoDevices[0].deviceId);
        } else {
          setSelectedCamera(videoDevices[0].deviceId);
        }
      }
    } catch (error) {
      console.error('Error refreshing cameras:', error);
      alert('שגיאה בגישה למצלמות. אנא וודא שיש גישה למצלמה.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!window.isSecureContext) {
      alert('האפליקציה צריכה לרוץ בסביבה מאובטחת. אנא השתמש ב-localhost או HTTPS');
      return;
    }

    refreshCameras();

    // Listen for device changes
    navigator.mediaDevices.addEventListener('devicechange', refreshCameras);
    
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', refreshCameras);
    };
  }, []);

  useEffect(() => {
    if (selectedCamera) {
      initCamera();
    }
  }, [selectedCamera]);

  // Add new effect to handle video size changes
  useEffect(() => {
    const handleVideoMetadata = () => {
      if (videoRef.current && canvasRef.current) {
        const width = videoRef.current.videoWidth;
        const height = videoRef.current.videoHeight;
        console.log('Video size:', { width, height });
        
        // Set canvas size to match video
        canvasRef.current.width = width;
        canvasRef.current.height = height;
        setCanvasSize({ width, height });
      }
    };

    if (videoRef.current) {
      videoRef.current.addEventListener('loadedmetadata', handleVideoMetadata);
      // Also handle the case where metadata is already loaded
      if (videoRef.current.videoWidth) {
        handleVideoMetadata();
      }
    }

    return () => {
      if (videoRef.current) {
        videoRef.current.removeEventListener('loadedmetadata', handleVideoMetadata);
      }
    };
  }, [videoRef.current]);

  // Scanning animation
  const animateScan = () => {
    setScanLine(prev => {
      if (!canvasRef.current) return 0;
      const newPos = prev + 2;
      return newPos >= canvasRef.current.height ? 0 : newPos;
    });
    scanAnimationRef.current = requestAnimationFrame(animateScan);
  };

  useEffect(() => {
    if (isDetecting) {
      animateScan();
    } else {
      cancelAnimationFrame(scanAnimationRef.current);
    }
    return () => cancelAnimationFrame(scanAnimationRef.current);
  }, [isDetecting]);

  const initSurface = () => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Draw current frame to canvas
    ctx.drawImage(videoRef.current, 0, 0);

    // Get image data from center of frame
    const centerX = canvasRef.current.width / 2;
    const centerY = canvasRef.current.height / 2;
    const sampleSize = 50;

    const imageData = ctx.getImageData(
      centerX - sampleSize/2,
      centerY - sampleSize/2,
      sampleSize,
      sampleSize
    );

    // Calculate average color
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < imageData.data.length; i += 4) {
      r += imageData.data[i];
      g += imageData.data[i + 1];
      b += imageData.data[i + 2];
    }

    const pixelCount = imageData.data.length / 4;
    const avgColor = {
      r: Math.round(r / pixelCount),
      g: Math.round(g / pixelCount),
      b: Math.round(b / pixelCount)
    };

    setSurfaceColors([avgColor]);
    setIsDetecting(false);
  };

  const handleSurfaceSelection = (event: React.MouseEvent | React.TouchEvent) => {
    console.log('Surface selection clicked!', { isSelectingSurface });
    if (!isSelectingSurface || !canvasRef.current) {
      console.log('Cannot select surface:', { 
        isSelectingSurface, 
        hasCanvas: !!canvasRef.current,
        event: event.type 
      });
      return;
    }

    event.preventDefault();

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Get coordinates
    let x: number, y: number;
    if ('touches' in event) {
      x = event.touches[0].clientX - rect.left;
      y = event.touches[0].clientY - rect.top;
    } else {
      x = event.clientX - rect.left;
      y = event.clientY - rect.top;
    }

    // Scale coordinates to canvas size
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    x = x * scaleX;
    y = y * scaleY;

    console.log('Click coordinates:', {
      original: { x, y },
      rect,
      scale: { x: scaleX, y: scaleY },
      canvas: { width: canvas.width, height: canvas.height }
    });

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      console.log('No canvas context!');
      return;
    }

    // Sample color around touch point
    const sampleSize = 30;
    const imageData = ctx.getImageData(
      Math.max(0, x - sampleSize/2),
      Math.max(0, y - sampleSize/2),
      sampleSize,
      sampleSize
    );

    // Calculate average color
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < imageData.data.length; i += 4) {
      r += imageData.data[i];
      g += imageData.data[i + 1];
      b += imageData.data[i + 2];
    }

    const pixelCount = imageData.data.length / 4;
    const avgColor = {
      r: Math.round(r / pixelCount),
      g: Math.round(g / pixelCount),
      b: Math.round(b / pixelCount)
    };

    console.log('Selected color:', avgColor);

    // Add color samples with variations
    const colors = [
      avgColor,
      // Darker variations
      {
        r: Math.max(0, avgColor.r - 30),
        g: Math.max(0, avgColor.g - 30),
        b: Math.max(0, avgColor.b - 30)
      },
      {
        r: Math.max(0, avgColor.r - 15),
        g: Math.max(0, avgColor.g - 15),
        b: Math.max(0, avgColor.b - 15)
      },
      // Lighter variations
      {
        r: Math.min(255, avgColor.r + 15),
        g: Math.min(255, avgColor.g + 15),
        b: Math.min(255, avgColor.b + 15)
      },
      {
        r: Math.min(255, avgColor.r + 30),
        g: Math.min(255, avgColor.g + 30),
        b: Math.min(255, avgColor.b + 30)
      }
    ];

    // First update the surface position
    setSelectedSurface({ x, y });
    console.log('Selected surface at:', { x, y });

    // Then update colors and exit selection mode
    setSurfaceColors(colors);
    setIsSelectingSurface(false);
  };

  const drawRegion = (region: Region) => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    
    ctx.strokeStyle = 'green';
    ctx.lineWidth = 2;
    ctx.strokeRect(region.minX, region.minY, region.maxX - region.minX, region.maxY - region.minY);
    
    // Draw the color sample
    ctx.fillStyle = `rgb(${region.color.r}, ${region.color.g}, ${region.color.b})`;
    ctx.fillRect(region.minX - 20, region.minY - 20, 15, 15);
  };

  return (
    <main className="fixed inset-0 flex items-center justify-center bg-gray-900 text-white">
      {/* Camera View */}
      <div className="relative w-full h-full">
        <video 
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover"
          playsInline
          autoPlay
        />
        <canvas 
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          onClick={handleSurfaceSelection}
          onTouchStart={handleSurfaceSelection}
          style={{ zIndex: 10, pointerEvents: 'auto' }}
        />
        {isRealTimeDetection && isProcessing && (
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-black/50 text-white px-4 py-2 rounded-full">
            מזהה...
          </div>
        )}
      </div>

      {/* Main Action Buttons */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-20 flex flex-col gap-2 w-[calc(100%-2rem)] max-w-sm">
        {isMainMenu ? (
          <>
            <button
              onClick={startManualDetection}
              className="w-full p-3 rounded-lg bg-green-500 hover:bg-green-600 font-semibold text-lg"
            >
              צלם וזהה גרביים
            </button>
            <button
              onClick={startRealTimeDetection}
              className="w-full p-3 rounded-lg bg-blue-500 hover:bg-blue-600 font-semibold text-lg"
            >
              זיהוי בזמן אמת
            </button>
          </>
        ) : isRealTimeDetection ? (
          <>
            <button
              onClick={returnToMainMenu}
              className="w-full p-3 rounded-lg bg-yellow-500 hover:bg-yellow-600 font-semibold text-lg"
              disabled={isProcessing}
            >
              חזור
            </button>
          </>
        ) : (
          <>
            {!capturedFrame ? (
              <button
                onClick={captureAndAnalyzeFrame}
                className="w-full p-3 rounded-lg bg-green-500 hover:bg-green-600 font-semibold text-lg"
              >
                צלם וזהה
              </button>
            ) : (
              <>
                <button
                  onClick={() => setCapturedFrame(false)}
                  className="w-full p-3 rounded-lg bg-green-500 hover:bg-green-600 font-semibold text-lg"
                >
                  צלם שוב
                </button>
                <button
                  onClick={returnToMainMenu}
                  className="w-full p-3 rounded-lg bg-yellow-500 hover:bg-yellow-600 font-semibold text-lg"
                >
                  חזור
                </button>
              </>
            )}
          </>
        )}
      </div>

      {/* Settings Button */}
      <button
        onClick={() => setShowSettings(!showSettings)}
        className="absolute top-4 right-4 z-10 p-2 rounded-full bg-gray-800/50 hover:bg-gray-700/50 transition-colors"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </button>

      {/* Settings Panel */}
      {showSettings && (
        <div className="absolute top-16 right-4 z-10 p-4 rounded-lg bg-gray-800/90 backdrop-blur-sm w-[calc(100%-2rem)] max-w-sm">
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">הגדרות</h2>
            
            {/* Camera Selection */}
            <div className="space-y-2">
              <label className="block text-sm">בחר מצלמה:</label>
              <div className="flex gap-2">
                <select
                  value={selectedCamera}
                  onChange={(e) => setSelectedCamera(e.target.value)}
                  className="flex-1 p-2 rounded bg-gray-700"
                  disabled={isLoading}
                >
                  {cameras.map(camera => (
                    <option key={camera.deviceId} value={camera.deviceId}>
                      {camera.label || `מצלמה ${cameras.indexOf(camera) + 1}`}
                    </option>
                  ))}
                </select>
                <button
                  onClick={refreshCameras}
                  className="p-2 rounded bg-blue-500 hover:bg-blue-600"
                  disabled={isLoading}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Visual Settings */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm mb-1">עובי קווים: {lineWidth}px</label>
                <input
                  type="range"
                  min="2"
                  max="8"
                  value={lineWidth}
                  onChange={(e) => setLineWidth(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm mb-1">גודל טקסט: {fontSize}px</label>
                <input
                  type="range"
                  min="16"
                  max="36"
                  value={fontSize}
                  onChange={(e) => setFontSize(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm mb-1">רגישות צבע: {settings.colorThreshold}</label>
                <input
                  type="range"
                  min="20"
                  max="50"
                  value={settings.colorThreshold}
                  onChange={(e) => setSettings({...settings, colorThreshold: parseInt(e.target.value)})}
                  className="w-full"
                />
                <span className="text-xs text-gray-400">ערך נמוך = רגישות גבוהה לשינויי צבע</span>
              </div>

              <div>
                <label className="block text-sm mb-1">גודל מינימלי: {settings.minRegionSize}</label>
                <input
                  type="range"
                  min="3000"
                  max="10000"
                  step="1000"
                  value={settings.minRegionSize}
                  onChange={(e) => setSettings({...settings, minRegionSize: parseInt(e.target.value)})}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm mb-1">גודל מקסימלי: {settings.maxRegionSize}</label>
                <input
                  type="range"
                  min="20000"
                  max="100000"
                  step="5000"
                  value={settings.maxRegionSize}
                  onChange={(e) => setSettings({...settings, maxRegionSize: parseInt(e.target.value)})}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
} 