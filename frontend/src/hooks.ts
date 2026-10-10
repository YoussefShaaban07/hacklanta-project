import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Poll an async function on an interval.
 *
 * The prototype polls once a second rather than adding a socket layer, as the
 * build plan suggests. Two details matter: a slow response must not stack up
 * overlapping requests, and a result that arrives after unmount must not set
 * state.
 */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  enabled = true,
) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const alive = useRef(true)
  const inFlight = useRef(false)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  const run = useCallback(async () => {
    if (inFlight.current) return
    inFlight.current = true
    try {
      const result = await fetcherRef.current()
      if (alive.current) {
        setData(result)
        setError(null)
      }
    } catch (err) {
      if (alive.current) setError(err instanceof Error ? err.message : String(err))
    } finally {
      inFlight.current = false
      if (alive.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    alive.current = true
    if (!enabled) {
      setLoading(false)
      return
    }
    run()
    const id = setInterval(run, intervalMs)
    return () => {
      alive.current = false
      clearInterval(id)
    }
  }, [run, intervalMs, enabled])

  return { data, error, loading, refresh: run }
}

export function useNow(intervalMs = 1000) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])
  return now
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return String(iso)
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function secondsBetween(a: string, b: string): string {
  const diff = (new Date(b).getTime() - new Date(a).getTime()) / 1000
  return Number.isFinite(diff) ? `${diff.toFixed(1)}s` : '—'
}
