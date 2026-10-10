import { useState } from 'react'
import { api } from '../api'
import { formatTime, usePolling } from '../hooks'
import type { DisplayState, HistoryAnswer, Workflow } from '../types'

interface Props {
  workflow: Workflow | null
  onChanged: () => Promise<unknown>
}

/** Live preview and inference status are deliberately separate indicators. */
const STATE_TONE: Record<DisplayState, string> = {
  monitoring: 'ok',
  confirming: 'warn',
  'event active': 'bad',
  unverified: 'unknown',
  uncertain: 'unknown',
  disconnected: 'bad',
  stale: 'warn',
  paused: '',
  starting: 'info',
  'not running': 'bad',
}

export function Monitor({ workflow, onChanged }: Props) {
  const [busy, setBusy] = useState(false)
  const [question, setQuestion] = useState('When did this rule last trigger?')
  const [answer, setAnswer] = useState<HistoryAnswer | null>(null)

  const { data: status } = usePolling(
    () => api.status(workflow!.id),
    1000,
    Boolean(workflow),
  )
  const { data: cameras } = usePolling(api.cameras, 5000)
  const camera = cameras?.[0]

  if (!workflow) {
    return (
      <div className="card">
        <div className="empty">
          No rule yet. Open <strong>Rule setup</strong> to describe what the
          camera should watch for.
        </div>
      </div>
    )
  }

  const toggle = async () => {
    setBusy(true)
    try {
      await api.patchWorkflow(workflow.id, { enabled: !workflow.enabled })
      await onChanged()
    } finally {
      setBusy(false)
    }
  }

  const ask = async () => {
    try {
      setAnswer(await api.askHistory(question, workflow.id))
    } catch (err) {
      setAnswer({
        supported: false, kind: 'error',
        answer: err instanceof Error ? err.message : String(err),
        events: [], actions: [], scope_note: '', suggestions: [],
      })
    }
  }

  const observation = status?.latest_observation
  const tone = status ? STATE_TONE[status.display_state] ?? '' : ''

  return (
    <div className="grid two">
      <div className="card">
        <div className="spread" style={{ marginBottom: 10 }}>
          <h2 style={{ margin: 0 }}>Live feed</h2>
          {camera && (
            <span className={`pill ${camera.is_live ? 'ok' : 'warn'}`}>
              {camera.is_live ? 'LIVE' : 'REPLAY'} · {camera.source}
            </span>
          )}
        </div>

        {/* A recorded or synthetic source is never presented as live. */}
        {camera && !camera.is_live && (
          <div className="banner warn small">
            This is recorded or synthetic footage, not a live camera. Event
            timing is relative to the source.
          </div>
        )}

        <div className="region-wrap">
          <img src={api.previewUrl(workflow.camera_id)} alt="camera preview" />
          <div
            className="region-box"
            style={{
              left: `${workflow.roi.x * 100}%`,
              top: `${workflow.roi.y * 100}%`,
              width: `${workflow.roi.w * 100}%`,
              height: `${workflow.roi.h * 100}%`,
            }}
          />
        </div>
        <div className="region-hint">
          Highlighted region is what the model inspects. Frame{' '}
          <span className="mono">{status?.camera.frame_id ?? '—'}</span>, age{' '}
          <span className="mono">{status?.camera.age_seconds ?? '—'}s</span>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <h2>Status</h2>
          <div className="row" style={{ marginBottom: 12 }}>
            <span className={`pill ${tone}`}>
              <span className="dot" />
              {status?.display_state ?? 'loading'}
            </span>
            <span className="pill">{status?.samples_taken ?? 0} samples</span>
            {status?.condition_unverified && (
              <span className="pill unknown">condition unverified</span>
            )}
          </div>

          <dl className="kv">
            <dt>Rule</dt>
            <dd>
              {workflow.target_description}{' '}
              <span className="muted small">v{workflow.version}</span>
            </dd>
            <dt>Confirmation</dt>
            <dd>
              {workflow.duration_seconds}s · ≥{workflow.min_samples} samples
            </dd>
            <dt>Progress</dt>
            <dd>
              {status?.positive_samples ?? 0} positive
              {status?.clear_streak ? ` · ${status.clear_streak} clearing` : ''}
            </dd>
            <dt>Action</dt>
            <dd>
              {workflow.action.type === 'create_task'
                ? `task: ${workflow.action.title}`
                : `webhook → ${workflow.action.destination_id}`}
            </dd>
          </dl>

          <div className="row" style={{ marginTop: 14 }}>
            <button
              className={workflow.enabled ? 'ghost' : 'primary'}
              onClick={toggle}
              disabled={busy}
            >
              {workflow.enabled ? 'Pause' : 'Activate'}
            </button>
          </div>
        </div>

        <div className="card">
          <h2>Latest observation</h2>
          {observation ? (
            <>
              <div className="row" style={{ marginBottom: 8 }}>
                <span
                  className={`pill ${
                    observation.result === 'met'
                      ? 'ok'
                      : observation.result === 'unknown'
                        ? 'unknown'
                        : ''
                  }`}
                >
                  {observation.result}
                </span>
                <span className="pill">{observation.inference_ms} ms</span>
                <span className="pill">{observation.model_id}</span>
              </div>
              <p className="small" style={{ margin: '0 0 6px' }}>
                {observation.explanation}
              </p>
              <div className="muted small">
                captured {formatTime(observation.captured_at_utc)}
              </div>
            </>
          ) : (
            <div className="empty">
              No observation yet.
              {!workflow.enabled && ' The rule is paused.'}
            </div>
          )}
        </div>

        <div className="card">
          <h2>Ask about history</h2>
          <div className="row">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && ask()}
            />
            <button className="ghost" onClick={ask}>
              Ask
            </button>
          </div>
          {answer && (
            <div style={{ marginTop: 10 }}>
              <div className={`banner ${answer.supported ? 'info' : 'warn'}`}>
                {answer.answer}
              </div>
              {answer.scope_note && (
                <div className="muted small">{answer.scope_note}</div>
              )}
              {answer.suggestions.length > 0 && (
                <div className="muted small" style={{ marginTop: 6 }}>
                  Try: {answer.suggestions.join(' · ')}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
