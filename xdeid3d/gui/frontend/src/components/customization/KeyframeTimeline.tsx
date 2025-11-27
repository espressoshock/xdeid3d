import { useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { useCustomization } from '@/hooks/useCustomization'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { X, Info } from 'lucide-react'
import { cn } from '@/lib/utils'

export function KeyframeTimeline() {
  const {
    cameraPose,
    setCurrentTime,
    setDuration,
    setInterpolation,
    updateKeyframe,
    removeKeyframe,
  } = useCustomization()

  // Extract duration and currentTime from cameraPose (they're nested, not top-level)
  const { duration, currentTime, keyframes, interpolation } = cameraPose

  const timelineRef = useRef<HTMLDivElement>(null)
  const [selectedKeyframeId, setSelectedKeyframeId] = useState<string | null>(null)

  const selectedKeyframe = keyframes.find(kf => kf.id === selectedKeyframeId)
  const hasInsufficientKeyframes = keyframes.length < 2

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current) return

    const rect = timelineRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const normalizedTime = Math.max(0, Math.min(1, x / rect.width))

    setCurrentTime(normalizedTime)
  }

  const handleKeyframeDragEnd = (keyframeId: string, newNormalizedTime: number) => {
    updateKeyframe(keyframeId, { timestamp: newNormalizedTime })
  }

  const formatTime = (normalizedTime: number) => {
    const seconds = normalizedTime * (duration || 10) // Fallback to 10 seconds
    return `${seconds.toFixed(1)}s`
  }

  return (
    <div className="space-y-4">
      {/* Helpful guide for keyframe mode */}
      {hasInsufficientKeyframes && (
        <Alert>
          <Info className="h-4 w-4" />
          <AlertTitle>How to Add Keyframes</AlertTitle>
          <AlertDescription className="text-xs space-y-1">
            <p className="font-medium">You need at least 2 keyframes to create an animation.</p>
            <ol className="list-decimal list-inside space-y-0.5 mt-2">
              <li>Rotate the 3D view to your desired camera angle</li>
              <li>Click "Add Keyframe" in the 3D viewer panel</li>
              <li>Repeat to add more keyframes at different angles</li>
            </ol>
          </AlertDescription>
        </Alert>
      )}

      {/* Timeline Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium">Timeline</h3>
          <p className="text-xs text-muted-foreground">
            {keyframes.length} keyframe{keyframes.length !== 1 ? 's' : ''}
            {keyframes.length >= 2 && (
              <span className="ml-2 text-primary">
                • {interpolation === 'cubic' ? 'Smooth' : 'Linear'} interpolation
              </span>
            )}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2">
            <Label htmlFor="duration" className="text-xs">Duration:</Label>
            <Input
              id="duration"
              type="number"
              min="1"
              max="60"
              step="1"
              value={duration || 10}
              onChange={(e) => setDuration(parseFloat(e.target.value))}
              className="w-16 h-8 text-xs"
            />
            <span className="text-xs text-muted-foreground">sec</span>
          </div>
        </div>
      </div>

      {/* Timeline Visualization */}
      <Card className="p-4">
        <div className="space-y-3">
          {/* Time indicator */}
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>0.0s</span>
            <span className="font-medium text-foreground">{formatTime(currentTime || 0)}</span>
            <span>{(duration || 10).toFixed(1)}s</span>
          </div>

          {/* Timeline track */}
          <div
            ref={timelineRef}
            className="relative h-12 bg-secondary/50 rounded-md cursor-pointer border border-border hover:border-primary/50 transition-colors"
            onClick={handleTimelineClick}
          >
            {/* Background grid lines */}
            <div className="absolute inset-0 flex">
              {[...Array(10)].map((_, i) => (
                <div
                  key={i}
                  className="flex-1 border-r border-border/30 last:border-r-0"
                />
              ))}
            </div>

            {/* Current time indicator */}
            <motion.div
              className="absolute top-0 bottom-0 w-0.5 bg-primary pointer-events-none"
              style={{ left: `${(currentTime || 0) * 100}%` }}
              initial={false}
              animate={{ left: `${(currentTime || 0) * 100}%` }}
              transition={{ type: 'spring', stiffness: 500, damping: 30 }}
            >
              <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-full">
                <div className="w-2 h-2 bg-primary rounded-full" />
              </div>
            </motion.div>

            {/* Keyframe markers */}
            {keyframes.map((keyframe) => (
              <KeyframeMarker
                key={keyframe.id}
                keyframe={keyframe}
                duration={duration || 10}
                isSelected={selectedKeyframeId === keyframe.id}
                onSelect={() => setSelectedKeyframeId(keyframe.id)}
                onDragEnd={(newTime) => handleKeyframeDragEnd(keyframe.id, newTime)}
                onRemove={() => removeKeyframe(keyframe.id)}
                timelineRef={timelineRef}
              />
            ))}
          </div>

          {/* Timeline controls */}
          <div className="flex items-center justify-between gap-2">
            {/* Interpolation mode toggle */}
            {keyframes.length >= 2 ? (
              <div className="flex items-center gap-1">
                <Label className="text-xs text-muted-foreground mr-1">Interpolation:</Label>
                <Button
                  size="sm"
                  variant={interpolation === 'linear' ? 'default' : 'ghost'}
                  onClick={() => setInterpolation('linear')}
                  className="h-7 px-2 text-xs"
                >
                  Linear
                </Button>
                <Button
                  size="sm"
                  variant={interpolation === 'cubic' ? 'default' : 'ghost'}
                  onClick={() => setInterpolation('cubic')}
                  className="h-7 px-2 text-xs"
                >
                  Smooth
                </Button>
              </div>
            ) : (
              <div className="text-xs text-muted-foreground">
                Add keyframes using the 3D viewer
              </div>
            )}

            {/* Keyframe count indicator */}
            <div className="text-xs text-muted-foreground">
              {keyframes.length} keyframe{keyframes.length !== 1 ? 's' : ''}
            </div>
          </div>
        </div>
      </Card>

      {/* Selected Keyframe Properties */}
      {selectedKeyframe && (
        <Card className="p-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Keyframe Properties</h4>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setSelectedKeyframeId(null)}
              >
                <X className="h-3.5 w-3.5" />
              </Button>
            </div>

            <Separator />

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label htmlFor="kf-time" className="text-xs">Time</Label>
                <Input
                  id="kf-time"
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={selectedKeyframe.timestamp.toFixed(2)}
                  onChange={(e) =>
                    updateKeyframe(selectedKeyframe.id, {
                      timestamp: parseFloat(e.target.value),
                    })
                  }
                  className="h-8 text-xs mt-1"
                />
                <span className="text-xs text-muted-foreground">
                  {formatTime(selectedKeyframe.timestamp)}
                </span>
              </div>

              <div>
                <Label htmlFor="kf-yaw" className="text-xs">Yaw</Label>
                <Input
                  id="kf-yaw"
                  type="number"
                  step="0.1"
                  value={(selectedKeyframe.yaw * 180 / Math.PI).toFixed(1)}
                  onChange={(e) =>
                    updateKeyframe(selectedKeyframe.id, {
                      yaw: (parseFloat(e.target.value) * Math.PI) / 180,
                    })
                  }
                  className="h-8 text-xs mt-1"
                />
                <span className="text-xs text-muted-foreground">degrees</span>
              </div>

              <div>
                <Label htmlFor="kf-pitch" className="text-xs">Pitch</Label>
                <Input
                  id="kf-pitch"
                  type="number"
                  step="0.1"
                  value={(selectedKeyframe.pitch * 180 / Math.PI).toFixed(1)}
                  onChange={(e) =>
                    updateKeyframe(selectedKeyframe.id, {
                      pitch: (parseFloat(e.target.value) * Math.PI) / 180,
                    })
                  }
                  className="h-8 text-xs mt-1"
                />
                <span className="text-xs text-muted-foreground">degrees</span>
              </div>

              <div>
                <Label htmlFor="kf-radius" className="text-xs">Distance</Label>
                <Input
                  id="kf-radius"
                  type="number"
                  min="1"
                  max="10"
                  step="0.1"
                  value={selectedKeyframe.radius.toFixed(1)}
                  onChange={(e) =>
                    updateKeyframe(selectedKeyframe.id, {
                      radius: parseFloat(e.target.value),
                    })
                  }
                  className="h-8 text-xs mt-1"
                />
                <span className="text-xs text-muted-foreground">units</span>
              </div>
            </div>

            <Button
              size="sm"
              variant="destructive"
              onClick={() => {
                removeKeyframe(selectedKeyframe.id)
                setSelectedKeyframeId(null)
              }}
              className="w-full mt-2"
            >
              <X className="h-3.5 w-3.5 mr-2" />
              Delete Keyframe
            </Button>
          </div>
        </Card>
      )}
    </div>
  )
}

interface KeyframeMarkerProps {
  keyframe: {
    id: string
    timestamp: number
    yaw: number
    pitch: number
    radius: number
  }
  duration: number
  isSelected: boolean
  onSelect: () => void
  onDragEnd: (newTime: number) => void
  onRemove: () => void
  timelineRef: React.RefObject<HTMLDivElement>
}

function KeyframeMarker({
  keyframe,
  duration,
  isSelected,
  onSelect,
  onDragEnd,
  onRemove,
  timelineRef,
}: KeyframeMarkerProps) {
  const handleDragEnd = (_: any, info: any) => {
    if (!timelineRef.current) return

    const rect = timelineRef.current.getBoundingClientRect()
    const newX = info.point.x - rect.left
    const normalizedTime = Math.max(0, Math.min(1, newX / rect.width))

    onDragEnd(normalizedTime)
  }

  return (
    <motion.div
      drag="x"
      dragConstraints={timelineRef}
      dragElastic={0}
      dragMomentum={false}
      dragPropagation={false}
      onDragEnd={handleDragEnd}
      onClick={(e) => {
        e.stopPropagation()
        onSelect()
      }}
      className={cn(
        'absolute top-1/2 -translate-y-1/2 -translate-x-1/2 cursor-grab active:cursor-grabbing z-10',
        'group'
      )}
      style={{ left: `${keyframe.timestamp * 100}%` }}
      whileHover={{ scale: 1.2 }}
      whileTap={{ scale: 0.9 }}
    >
      {/* Keyframe marker */}
      <div
        className={cn(
          'w-4 h-4 rounded-full border-2 transition-colors',
          isSelected
            ? 'bg-primary border-primary shadow-lg shadow-primary/50'
            : 'bg-green-500 border-green-600 hover:border-green-400'
        )}
      />

      {/* Vertical line */}
      <div
        className={cn(
          'absolute top-1/2 left-1/2 -translate-x-1/2 w-0.5 h-12 -translate-y-1/2 pointer-events-none',
          isSelected ? 'bg-primary/30' : 'bg-green-500/20'
        )}
      />

      {/* Enhanced tooltip on hover */}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-20">
        <div className="bg-popover text-popover-foreground text-xs px-3 py-2 rounded-md shadow-lg whitespace-nowrap border border-border">
          <div className="font-medium mb-1">{(keyframe.timestamp * duration).toFixed(2)}s</div>
          <div className="text-muted-foreground space-y-0.5">
            <div>Yaw: {(keyframe.yaw * 180 / Math.PI).toFixed(1)}°</div>
            <div>Pitch: {(keyframe.pitch * 180 / Math.PI).toFixed(1)}°</div>
            <div>Distance: {keyframe.radius.toFixed(1)}</div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
