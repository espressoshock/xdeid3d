import { Suspense, useRef, useState, useEffect, useCallback } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import { Loader2, Camera, Maximize2, Video } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import * as THREE from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'

interface ThreeDViewerProps {
  meshUrl: string
  onCaptureView?: (pose: { yaw: number; pitch: number; radius: number }) => void
}

interface MeshModelProps {
  url: string
  onGeometryLoaded?: (boundingSphere: THREE.Sphere) => void
}

function MeshModel({ url, onGeometryLoaded }: MeshModelProps) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loader = new PLYLoader()

    // Convert relative URL to absolute URL
    const absoluteUrl = url.startsWith('http') ? url : `http://localhost:8000${url}`

    console.log('[ThreeDViewer] Loading mesh from:', absoluteUrl)

    loader.load(
      absoluteUrl,
      (geometry) => {
        console.log('[ThreeDViewer] Mesh loaded successfully')
        // Center the geometry
        geometry.center()
        geometry.computeVertexNormals()

        // Compute bounding sphere for camera fitting
        geometry.computeBoundingSphere()

        setGeometry(geometry)
      },
      (progress) => {
        console.log('[ThreeDViewer] Loading progress:', (progress.loaded / progress.total) * 100, '%')
      },
      (error) => {
        console.error('[ThreeDViewer] Error loading mesh:', error)
        setError(error.message || 'Failed to load mesh')
      }
    )
  }, [url])

  // Notify parent when geometry loads
  useEffect(() => {
    if (geometry?.boundingSphere && onGeometryLoaded) {
      onGeometryLoaded(geometry.boundingSphere)
    }
  }, [geometry, onGeometryLoaded])

  if (error) {
    return (
      <mesh>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="red" />
      </mesh>
    )
  }

  if (!geometry) {
    return null
  }

  return (
    <mesh geometry={geometry} rotation={[0, 0, 0]} scale={0.5}>
      <meshStandardMaterial
        color="#E0E0E0"
        roughness={0.5}
        metalness={0.2}
        flatShading={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

function CameraHelper({
  cameraRef,
  onCameraMove
}: {
  cameraRef: React.MutableRefObject<THREE.PerspectiveCamera | null>
  onCameraMove?: (angles: { yaw: number; pitch: number; radius: number }) => void
}) {
  const { camera } = useThree()

  useEffect(() => {
    // Store camera ref for parent component
    if (camera instanceof THREE.PerspectiveCamera) {
      cameraRef.current = camera
    }
  }, [camera, cameraRef])

  useEffect(() => {
    const updateAngles = () => {
      // Calculate spherical coordinates from camera position
      const pos = camera.position
      const radius = Math.sqrt(pos.x ** 2 + pos.y ** 2 + pos.z ** 2)

      // Yaw (horizontal angle) - rotation around Y axis (in radians)
      const yaw = Math.atan2(pos.x, pos.z)

      // Pitch (vertical angle) - angle from Y axis (in radians)
      const pitch = Math.acos(pos.y / radius)

      // Notify parent component with updated angles (in radians)
      if (onCameraMove) {
        onCameraMove({ yaw, pitch, radius })
      }
    }

    // Update on camera movement
    const interval = setInterval(updateAngles, 100)
    return () => clearInterval(interval)
  }, [camera, onCameraMove])

  return null
}

interface SceneProps {
  meshUrl: string
  cameraRef: React.MutableRefObject<THREE.PerspectiveCamera | null>
  controlsRef: React.MutableRefObject<any>
  controlsLimits: { minDistance: number; maxDistance: number }
  onGeometryLoaded?: (boundingSphere: THREE.Sphere) => void
  onCameraMove?: (angles: { yaw: number; pitch: number; radius: number }) => void
}

function Scene({ meshUrl, cameraRef, controlsRef, controlsLimits, onGeometryLoaded, onCameraMove }: SceneProps) {
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <directionalLight position={[-10, -10, -5]} intensity={0.5} />

      {/* Mesh */}
      <Suspense fallback={null}>
        <MeshModel url={meshUrl} onGeometryLoaded={onGeometryLoaded} />
      </Suspense>

      {/* Grid floor */}
      <Grid
        args={[10, 10]}
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#6B7280"
        sectionSize={1}
        sectionThickness={1}
        sectionColor="#4B5563"
        fadeDistance={20}
        fadeStrength={1}
        position={[0, -1.5, 0]}
      />

      {/* Controls */}
      <OrbitControls
        ref={controlsRef}
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={controlsLimits.minDistance}
        maxDistance={controlsLimits.maxDistance}
        target={[0, 0, 0]}
      />

      {/* Camera angle tracker */}
      <CameraHelper cameraRef={cameraRef} onCameraMove={onCameraMove} />
    </>
  )
}

export function ThreeDViewer({ meshUrl, onCaptureView }: ThreeDViewerProps) {
  const controlsRef = useRef<any>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const hasFittedCameraRef = useRef(false)
  const previousMeshRadiusRef = useRef<number | null>(null)
  // Camera angles in radians (will be converted to degrees for display)
  const [cameraAngles, setCameraAngles] = useState({ yaw: 0, pitch: Math.PI / 2, radius: 5 })
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [controlsLimits, setControlsLimits] = useState({ minDistance: 1, maxDistance: 1000 })
  const [showFrameOverlay, setShowFrameOverlay] = useState(true)

  // Callback for when camera moves (receives angles in radians)
  const handleCameraMove = useCallback((angles: { yaw: number; pitch: number; radius: number }) => {
    setCameraAngles(angles)
  }, [])

  const fitCameraToMesh = useCallback((boundingSphere: THREE.Sphere, forceReposition: boolean = false) => {
    if (!cameraRef.current || !controlsRef.current) {
      console.warn('[ThreeDViewer] Camera or controls not ready for fitting')
      return
    }

    const camera = cameraRef.current
    const controls = controlsRef.current

    // Get mesh bounding sphere
    const center = boundingSphere.center
    const radius = boundingSphere.radius

    console.log('[ThreeDViewer] Fitting camera to mesh:', {
      center,
      radius,
      previousRadius: previousMeshRadiusRef.current,
      forceReposition
    })

    // Calculate camera distance based on FOV and mesh size
    // We want the mesh to fit in the view with padding
    const fov = camera.fov * (Math.PI / 180) // Convert to radians
    const paddingFactor = 1.25 // Add 25% padding around the mesh
    const fittedDistance = (radius * paddingFactor) / Math.tan(fov / 2)

    // Update OrbitControls distance limits based on mesh size
    // Allow zooming from 0.3x to 4x the fitted distance
    const newMinDistance = fittedDistance * 0.3
    const newMaxDistance = fittedDistance * 4

    console.log('[ThreeDViewer] Updating controls limits:', {
      minDistance: newMinDistance,
      maxDistance: newMaxDistance,
    })

    // Always update controls limits (needed for proper zoom on regeneration)
    controls.minDistance = newMinDistance
    controls.maxDistance = newMaxDistance
    setControlsLimits({ minDistance: newMinDistance, maxDistance: newMaxDistance })

    // Only reposition camera on first load, not on regeneration
    // On regeneration, scale camera distance to maintain visual padding
    if (!hasFittedCameraRef.current || forceReposition) {
      // FIRST LOAD: Position camera at an optimal aesthetic angle
      // 3/4 view with slight elevation for best presentation
      const yawDeg = 25   // Rotate 25 degrees to the side
      const pitchDeg = 80 // 80 degrees from Y-axis (10 degrees above horizon)

      const yawRad = yawDeg * (Math.PI / 180)
      const pitchRad = pitchDeg * (Math.PI / 180)

      // Convert spherical coordinates to Cartesian
      const x = fittedDistance * Math.sin(pitchRad) * Math.sin(yawRad)
      const y = fittedDistance * Math.cos(pitchRad)
      const z = fittedDistance * Math.sin(pitchRad) * Math.cos(yawRad)

      camera.position.set(x, y, z)

      // Update OrbitControls target to mesh center
      controls.target.set(center.x, center.y, center.z)

      // Mark that we've fitted the camera and store mesh radius
      hasFittedCameraRef.current = true
      previousMeshRadiusRef.current = radius

      console.log('[ThreeDViewer] Camera repositioned (first load):', {
        position: camera.position,
        target: controls.target,
        fittedDistance,
        angle: { yawDeg, pitchDeg },
      })
    } else if (previousMeshRadiusRef.current !== null) {
      // REGENERATION: Scale camera distance proportionally to maintain visual padding
      const previousRadius = previousMeshRadiusRef.current
      const scaleFactor = radius / previousRadius

      console.log('[ThreeDViewer] Scaling camera for regeneration:', {
        previousRadius,
        newRadius: radius,
        scaleFactor,
      })

      // Get current camera position
      const currentPos = camera.position.clone()

      // Get direction from origin to camera (preserves angle)
      const direction = currentPos.normalize()

      // Get current distance from origin
      const currentDistance = camera.position.length()

      // Calculate new distance by scaling current distance
      const newDistance = currentDistance * scaleFactor

      // Set new camera position (same direction, scaled distance)
      camera.position.copy(direction.multiplyScalar(newDistance))

      // Update OrbitControls target to mesh center
      controls.target.set(center.x, center.y, center.z)

      // Store new mesh radius for next regeneration
      previousMeshRadiusRef.current = radius

      console.log('[ThreeDViewer] Camera scaled for regeneration:', {
        oldDistance: currentDistance,
        newDistance,
        position: camera.position,
        visualPaddingPreserved: true,
      })
    }

    // Update controls
    controls.update()
  }, [])

  const handleGeometryLoaded = useCallback((boundingSphere: THREE.Sphere) => {
    console.log('[ThreeDViewer] Geometry loaded, auto-fitting camera')
    // Use a small delay to ensure camera and controls refs are populated
    setTimeout(() => {
      fitCameraToMesh(boundingSphere)
    }, 100)
  }, [fitCameraToMesh])

  const handleCaptureView = () => {
    if (onCaptureView) {
      // cameraAngles are already in radians, pass them directly
      const pose = {
        yaw: cameraAngles.yaw,
        pitch: cameraAngles.pitch,
        radius: cameraAngles.radius,
      }
      console.log('[ThreeDViewer] Capturing view:', {
        radians: pose,
        degrees: {
          yaw: (pose.yaw * 180 / Math.PI).toFixed(1),
          pitch: (pose.pitch * 180 / Math.PI).toFixed(1),
          radius: pose.radius.toFixed(2),
        }
      })
      onCaptureView(pose)
    }
  }

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen)
  }

  return (
    <div className="h-full w-full relative bg-[#1A1E37]">
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 0, 45], fov: 50 }}
        gl={{ antialias: true }}
        shadows
      >
        <Scene
          meshUrl={meshUrl}
          cameraRef={cameraRef}
          controlsRef={controlsRef}
          controlsLimits={controlsLimits}
          onGeometryLoaded={handleGeometryLoaded}
          onCameraMove={handleCameraMove}
        />
      </Canvas>

      {/* Camera Frame Overlay - Shows what will be rendered */}
      {showFrameOverlay && (
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
          {/* Render frame with 1:1 aspect ratio (512x512 typical render) */}
          <div className="relative" style={{ width: '70%', paddingBottom: '70%' }}>
            {/* Frame border */}
            <div className="absolute inset-0 border-2 border-primary/60 rounded-sm">
              {/* Corner markers */}
              <div className="absolute top-0 left-0 w-4 h-4 border-l-2 border-t-2 border-primary" />
              <div className="absolute top-0 right-0 w-4 h-4 border-r-2 border-t-2 border-primary" />
              <div className="absolute bottom-0 left-0 w-4 h-4 border-l-2 border-b-2 border-primary" />
              <div className="absolute bottom-0 right-0 w-4 h-4 border-r-2 border-b-2 border-primary" />

              {/* Rule of thirds grid */}
              <div className="absolute inset-0">
                {/* Vertical lines */}
                <div className="absolute left-1/3 top-0 bottom-0 w-px bg-primary/20" />
                <div className="absolute left-2/3 top-0 bottom-0 w-px bg-primary/20" />
                {/* Horizontal lines */}
                <div className="absolute top-1/3 left-0 right-0 h-px bg-primary/20" />
                <div className="absolute top-2/3 left-0 right-0 h-px bg-primary/20" />
              </div>

              {/* Center crosshair */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-6 h-px bg-primary/40" />
                <div className="absolute h-6 w-px bg-primary/40" />
              </div>

              {/* Safe area indicator */}
              <div className="absolute inset-4 border border-primary/20 border-dashed rounded-sm" />
            </div>

            {/* Frame label */}
            <div className="absolute -top-7 left-0 right-0 flex items-center justify-center gap-2 text-xs text-primary font-medium">
              <Video className="h-3 w-3" />
              <span>RENDER FRAME</span>
            </div>
          </div>

          {/* Frame toggle button (bottom left) */}
          <div className="absolute bottom-4 left-4 pointer-events-auto">
            <Button
              size="sm"
              variant={showFrameOverlay ? 'default' : 'outline'}
              onClick={() => setShowFrameOverlay(!showFrameOverlay)}
              className="gap-2 text-xs"
            >
              <Video className="h-3 w-3" />
              Frame Guide
            </Button>
          </div>
        </div>
      )}

      {/* Overlay UI */}
      <div className="absolute top-4 right-4 space-y-2">
        {/* Camera Info Card - Real-time updated */}
        <Card className="p-3 bg-background/95 backdrop-blur-sm border-border shadow-lg">
          <div className="space-y-1 text-xs">
            <div className="flex items-center gap-2 text-muted-foreground mb-2">
              <Camera className="h-3.5 w-3.5 text-primary" />
              <span className="font-medium">Camera Position</span>
              <span className="ml-auto text-[10px] text-primary">● LIVE</span>
            </div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
              <span className="text-muted-foreground">Yaw:</span>
              <span className="font-mono text-foreground font-medium tabular-nums">
                {((cameraAngles.yaw * 180) / Math.PI).toFixed(1)}°
              </span>
              <span className="text-muted-foreground">Pitch:</span>
              <span className="font-mono text-foreground font-medium tabular-nums">
                {((cameraAngles.pitch * 180) / Math.PI).toFixed(1)}°
              </span>
              <span className="text-muted-foreground">Distance:</span>
              <span className="font-mono text-foreground font-medium tabular-nums">
                {cameraAngles.radius.toFixed(2)}
              </span>
            </div>
          </div>
        </Card>

        {/* Action Buttons */}
        <div className="space-y-2">
          {onCaptureView && (
            <div className="space-y-1">
              <Button
                onClick={handleCaptureView}
                size="sm"
                className="w-full gap-2"
              >
                <Camera className="h-4 w-4" />
                Add Keyframe
              </Button>
              <p className="text-[10px] text-muted-foreground text-center px-1">
                Rotate the view, then click to capture
              </p>
            </div>
          )}

          <Button
            onClick={toggleFullscreen}
            size="sm"
            variant="outline"
            className="w-full"
          >
            <Maximize2 className="h-3.5 w-3.5 mr-2" />
            {isFullscreen ? 'Exit' : 'Fullscreen'}
          </Button>
        </div>
      </div>

      {/* Loading Overlay */}
      <Suspense
        fallback={
          <div className="absolute inset-0 flex items-center justify-center bg-[#1A1E37]">
            <div className="text-center">
              <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">Loading 3D mesh...</p>
            </div>
          </div>
        }
      >
        <div />
      </Suspense>

      {/* Help Text - Moved to bottom center to avoid frame toggle */}
      {!showFrameOverlay && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-xs text-muted-foreground text-center bg-background/80 backdrop-blur-sm px-3 py-2 rounded-md">
          <p className="flex items-center gap-1">
            <span className="text-primary">•</span> Left click + drag to rotate
            <span className="mx-2 text-border">|</span>
            <span className="text-primary">•</span> Right click + drag to pan
            <span className="mx-2 text-border">|</span>
            <span className="text-primary">•</span> Scroll to zoom
          </p>
        </div>
      )}
    </div>
  )
}
