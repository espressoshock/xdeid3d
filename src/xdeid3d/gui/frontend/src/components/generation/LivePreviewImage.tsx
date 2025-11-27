import { useState, useEffect } from 'react'

interface LivePreviewImageProps {
  src: string
  alt: string
}

/**
 * LivePreviewImage - Simple preview image component with instant updates
 *
 * No transitions or loading states - just instant replacement for seamless progression.
 */
export function LivePreviewImage({ src, alt }: LivePreviewImageProps) {
  const [hasError, setHasError] = useState(false)

  useEffect(() => {
    // Reset error state when src changes
    setHasError(false)
    console.log('[LivePreviewImage] Preview updated:', src)
  }, [src])

  if (hasError) {
    return (
      <div className="w-full aspect-square bg-muted flex items-center justify-center p-4">
        <div className="text-center text-sm text-muted-foreground">
          <p className="font-medium mb-1">Failed to load preview image</p>
          <p className="text-xs break-all">{src}</p>
        </div>
      </div>
    )
  }

  return (
    <img
      src={src}
      alt={alt}
      className="w-full h-full object-contain"
      onError={(e) => {
        console.error('[LivePreviewImage] Image element error:', e)
        setHasError(true)
      }}
    />
  )
}
