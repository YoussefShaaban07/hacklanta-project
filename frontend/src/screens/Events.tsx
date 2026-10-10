import { useState } from 'react'
import { api } from '../api'
import { formatTime, secondsBetween, usePolling } from '../hooks'
import type { ActionRecord, CameraEvent } from '../types'

interface Props {
  workflowId: string | null
}

function ActionRow({ action, onRetried }: { action: ActionRecord; onRetried: () => void }) {
  const [busy, setBusy] = useState(false)
  const tone =
    action.status === 'succeeded' ? 'ok' : action.status === 'failed' ? 'bad' : 'warn'

  const retry = async () => {
    setBusy(true)
    try {
      await api.retryAction(action.id)
      onRetried()
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="row" style={{ marginTop: 8 }}>
      <span className={`pill ${tone}`}>
        <span className="dot" />
        {action.type} {action.status}
      </span>
      <span className="pill">{action.attempts} attempt(s)</span>
      {action.receipt && (
        <span className="mono muted" style={{ flex: 1, minWidth: 180 }}>
          {action.receipt.slice(0, 90)}
        </span>
      )}
      {/* A failed delivery is shown as failed, never hidden. */}
      {action.status === 'failed' && (
        <>
          <span className="mono" style={{ color: 'var(--bad)' }}>
            {action.last_error?.slice(0, 70)}
          </span>
          <button className="danger" onClick={retry} disabled={busy}>
            Retry
          </button>
        </>
      )}
    </div>
  )
}

function EventCard({ event, onChanged }: { event: CameraEvent; onChanged: () => void }) {
  const snapshot = api.evidenceUrl(event.snapshot_path)
  const clip = api.evidenceUrl(event.clip_path)
  const tone =
    event.status === 'active' ? 'bad' : event.status === 'unverified' ? 'unknown' : ''

  return (
    <li className="card">
      <div className="spread">
        <h3>{event.explanation || 'Condition confirmed'}</h3>
        <span className={`pill ${tone}`}>{event.status}</span>
      </div>

      <dl className="kv" style={{ marginTop: 8 }}>
        <dt>Triggered</dt>
        <dd>{formatTime(event.triggered_at)}</dd>
        <dt>First positive</dt>
        <dd>
          {formatTime(event.first_positive_at)}{' '}
          <span className="muted small">
            (held {secondsBetween(event.first_positive_at, event.triggered_at)})
          </span>
        </dd>
        <dt>Rule</dt>
        <dd className="mono">
          {event.workflow_id} v{event.workflow_version}
        </dd>
        <dt>Episode</dt>
        <dd className="mono">{event.episode_id}</dd>
      </dl>

      {event.actions.map((action) => (
        <ActionRow key={action.id} action={action} onRetried={onChanged} />
      ))}

      <div className="evidence">
        {snapshot ? (
          <figure style={{ margin: 0 }}>
            <img src={snapshot} alt="trigger snapshot" />
            <figcaption className="muted small">Trigger snapshot</figcaption>
          </figure>
        ) : (
          <div className="empty small">No snapshot</div>
        )}
        {clip ? (
          <figure style={{ margin: 0 }}>
            <video src={clip} controls />
            <figcaption className="muted small">
              Clip around the trigger
            </figcaption>
          </figure>
        ) : (
          <div className="empty small">Clip processing…</div>
        )}
      </div>
    </li>
  )
}

export function Events({ workflowId }: Props) {
  const { data: events, refresh } = usePolling(
    () => api.events(workflowId ?? undefined, 50),
    2000,
  )

  if (!events) return <div className="card"><div className="empty">Loading…</div></div>

  if (events.length === 0) {
    return (
      <div className="card">
        <div className="empty">
          No recorded events yet. That is not the same as nothing having
          happened — it means nothing matched while the rule was active.
        </div>
      </div>
    )
  }

  return (
    <ul className="clean">
      {events.map((event) => (
        <EventCard key={event.id} event={event} onChanged={refresh} />
      ))}
    </ul>
  )
}
