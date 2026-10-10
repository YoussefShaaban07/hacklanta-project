import { useCallback, useRef, useState } from 'react'
import type { ROI } from '../types'

/**
 * Draw one normalized region over a camera frame.
 *
 * The subtlety the build plan calls out: the <img> is letterboxed inside its
 * box, so the element's rectangle is NOT the image's rectangle. Using
 * getBoundingClientRect() directly would skew every coordinate whenever the
 * frame's aspect ratio differs from the container's. We therefore compute the
 * displayed content rect from naturalWidth/naturalHeight and map pointer
 * positions into that, which keeps the drawn region aligned with what the
 * model will actually be given.
 */

interface Props {
  src: string
  roi: ROI | null
  onChange: (roi: ROI) => void
  disabled?: boolean
}

interface ContentRect {
  left: number
  top: number
  width: number
  height: number
}

function contentRect(img: HTMLImageElement): ContentRect | null {
  const { naturalWidth: nw, naturalHeight: nh } = img
  if (!nw || !nh) return null
  const box = img.getBoundingClientRect()
  // The image is rendered with width:100% and auto height, so it fills the
  // box horizontally. Computing it generally keeps this correct if the CSS
  // later switches to object-fit: contain.
  const scale = Math.min(box.width / nw, box.height / nh)
  const width = nw * scale
  const height = nh * scale
  return {
    left: box.left + (box.width - width) / 2,
    top: box.top + (box.height - height) / 2,
    width,
    height,
  }
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v))

export function RegionSelector({ src, roi, onChange, disabled }: Props) {
  const imgRef = useRef<HTMLImageElement>(null)
  const startRef = useRef<{ x: number; y: number } | null>(null)
  const [dragging, setDragging] = useState(false)

  const toNormalized = useCallback((clientX: number, clientY: number) => {
    const img = imgRef.current
    if (!img) return null
    const rect = contentRect(img)
    if (!rect) return null
    return {
      x: clamp01((clientX - rect.left) / rect.width),
      y: clamp01((clientY - rect.top) / rect.height),
    }
  }, [])

  const emit = useCallback(
    (a: { x: number; y: number }, b: { x: number; y: number }) => {
      const x = Math.min(a.x, b.x)
      const y = Math.min(a.y, b.y)
      const w = Math.abs(a.x - b.x)
      const h = Math.abs(a.y - b.y)
      // The backend rejects a zero-area or out-of-bounds region, so clamp to a
      // minimum here rather than sending something that will 422.
      onChange({
        x: Number(x.toFixed(4)),
        y: Number(y.toFixed(4)),
        w: Number(Math.max(0.02, Math.min(1 - x, w)).toFixed(4)),
        h: Number(Math.max(0.02, Math.min(1 - y, h)).toFixed(4)),
      })
    },
    [onChange],
  )

  const onPointerDown = (e: React.PointerEvent) => {
    if (disabled) return
    const point = toNormalized(e.clientX, e.clientY)
    if (!point) return
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    startRef.current = point
    setDragging(true)
    emit(point, point)
  }

  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging || !startRef.current) return
    const point = toNormalized(e.clientX, e.clientY)
    if (point) emit(startRef.current, point)
  }

  const endDrag = () => {
    startRef.current = null
    setDragging(false)
  }

  const style = roi
    ? {
        left: `${roi.x * 100}%`,
        top: `${roi.y * 100}%`,
        width: `${roi.w * 100}%`,
        height: `${roi.h * 100}%`,
      }
    : undefined

  return (
    <div>
      <div
        className="region-wrap"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        <img ref={imgRef} src={src} alt="camera frame" draggable={false} />
        {roi && <div className="region-box" style={style} />}
      </div>
      <div className="region-hint">
        {disabled
          ? 'Region is locked while the rule is active. Pause it to redraw.'
          : 'Drag on the frame to draw the region the model will inspect.'}
        {roi && (
          <span className="mono">
            {' '}
            x={roi.x.toFixed(2)} y={roi.y.toFixed(2)} w={roi.w.toFixed(2)} h=
            {roi.h.toFixed(2)}
          </span>
        )}
      </div>
    </div>
  )
}
