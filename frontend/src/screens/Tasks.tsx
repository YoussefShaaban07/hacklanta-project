import { api } from '../api'
import { formatTime, usePolling } from '../hooks'

export function Tasks() {
  const { data: tasks } = usePolling(api.tasks, 2000)

  if (!tasks) return <div className="card"><div className="empty">Loading…</div></div>

  if (tasks.length === 0) {
    return (
      <div className="card">
        <div className="empty">
          No tasks yet. A task is created when a rule's condition is confirmed.
        </div>
      </div>
    )
  }

  const open = tasks.filter((t) => !t.completed)
  const done = tasks.filter((t) => t.completed)

  return (
    <div className="grid">
      <div className="card">
        <h2>Open ({open.length})</h2>
        {open.length === 0 ? (
          <div className="empty small">Nothing open.</div>
        ) : (
          <ul className="clean">
            {open.map((task) => {
              const evidence = api.evidenceUrl(task.evidence_path)
              return (
                <li key={task.id} className="card" style={{ background: 'var(--panel-2)' }}>
                  <div className="spread">
                    <h3>{task.title}</h3>
                    <span className="pill warn">open</span>
                  </div>
                  <div className="muted small">
                    created {formatTime(task.created_at)} · rule{' '}
                    <span className="mono">{task.workflow_id}</span> · event{' '}
                    <span className="mono">{task.event_id}</span>
                  </div>
                  {evidence && (
                    <img
                      src={evidence}
                      alt="task evidence"
                      style={{
                        marginTop: 10, maxWidth: 320, width: '100%',
                        borderRadius: 8, border: '1px solid var(--line)',
                      }}
                    />
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </div>

      {done.length > 0 && (
        <div className="card">
          <h2>Completed ({done.length})</h2>
          <ul className="clean">
            {done.map((task) => (
              <li key={task.id} className="row">
                <span className="pill ok">done</span>
                <span>{task.title}</span>
                <span className="muted small">{formatTime(task.created_at)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
