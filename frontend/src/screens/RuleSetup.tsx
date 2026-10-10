import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { RegionSelector } from '../components/RegionSelector'
import type { CompileResponse, ROI, Workflow } from '../types'

interface Props {
  workflow: Workflow | null
  destinations: string[]
  onSaved: (id: string) => Promise<void> | void
  onChanged: () => Promise<unknown>
}

const EXAMPLE = 'If the supply tray stays empty for 10 seconds, create a restocking task.'

/**
 * Four connected cards in a fixed sequence: Camera, Visual condition,
 * Temporal confirmation, Software action. Editing any card updates the
 * validated draft; nothing is ever activated without an explicit click.
 */
export function RuleSetup({ workflow, destinations, onSaved, onChanged }: Props) {
  const [instruction, setInstruction] = useState(EXAMPLE)
  const [result, setResult] = useState<CompileResponse | null>(null)
  const [draft, setDraft] = useState<Workflow | null>(null)
  const [frameSrc, setFrameSrc] = useState<string>('')
  const [cropSrc, setCropSrc] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // One still frame to draw on. A frozen frame is the right surface for
  // drawing; the moving preview lives on the Monitor screen.
  const grabFrame = useCallback(() => {
    setFrameSrc(api.snapshotUrl(workflow?.camera_id ?? 'cam_01'))
  }, [workflow?.camera_id])

  useEffect(grabFrame, [grabFrame])

  const compile = async () => {
    setBusy(true)
    setError(null)
    try {
      const response = await api.compile(
        instruction,
        workflow?.camera_id ?? 'cam_01',
        draft?.roi,
      )
      setResult(response)
      setDraft(response.draft)
      setCropSrc(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const update = (patch: Partial<Workflow>) =>
    setDraft((current) => (current ? { ...current, ...patch } : current))

  const setRoi = (roi: ROI) => {
    update({ roi })
    setCropSrc(null)
  }

  const showCrop = async () => {
    if (!draft) return
    try {
      setCropSrc(await api.cropPreview(draft.camera_id, draft.roi))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const activate = async (enabled: boolean) => {
    if (!draft) return
    setBusy(true)
    setError(null)
    try {
      const saved = await api.saveWorkflow({ ...draft, enabled })
      await onChanged()
      await onSaved(saved.id)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  // A user-defined visual condition must pass a preview before activation.
  const previewBlocked =
    Boolean(draft) &&
    draft!.condition_type === 'visual_condition' &&
    !draft!.preview_validated

  return (
    <div className="grid two">
      <div className="grid">
        <div className="card">
          <h2>1 · Describe the task</h2>
          <textarea
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            placeholder={EXAMPLE}
          />
          <div className="row" style={{ marginTop: 10 }}>
            <button className="primary" onClick={compile} disabled={busy}>
              Review workflow
            </button>
            <button className="ghost" onClick={() => setInstruction(EXAMPLE)}>
              Use example
            </button>
          </div>

          {error && <div className="banner bad" style={{ marginTop: 10 }}>{error}</div>}

          {result && !result.ok && result.unsupported_reason && (
            <div className="banner warn" style={{ marginTop: 10 }}>
              <strong>Not supported.</strong> {result.unsupported_reason}
            </div>
          )}

          {result && !result.ok && result.questions.length > 0 && (
            <div className="banner info" style={{ marginTop: 10 }}>
              <strong>Needs more detail:</strong>
              <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                {result.questions.map((q) => (
                  <li key={q}>{q}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="card">
          <h2>2 · Camera and region</h2>
          <RegionSelector
            src={frameSrc}
            roi={draft?.roi ?? workflow?.roi ?? null}
            onChange={setRoi}
            disabled={!draft}
          />
          <div className="row" style={{ marginTop: 10 }}>
            <button className="ghost" onClick={grabFrame}>
              Refresh frame
            </button>
            <button className="ghost" onClick={showCrop} disabled={!draft}>
              Preview exact crop
            </button>
          </div>
          {cropSrc && (
            <div style={{ marginTop: 10 }}>
              <div className="muted small" style={{ marginBottom: 6 }}>
                This is precisely what the model receives:
              </div>
              <img
                src={cropSrc}
                alt="crop preview"
                style={{ maxWidth: '100%', borderRadius: 8, border: '1px solid var(--line)' }}
              />
            </div>
          )}
        </div>
      </div>

      <div className="grid">
        {draft ? (
          <>
            <div className="card">
              <h2>3 · Visual condition</h2>
              <label className="field">
                <span>What the model looks for</span>
                <input
                  type="text"
                  value={draft.target_description}
                  onChange={(e) => update({ target_description: e.target.value })}
                />
              </label>
              <label className="field">
                <span>Condition template</span>
                <select
                  value={draft.condition_type}
                  onChange={(e) =>
                    update({ condition_type: e.target.value as Workflow['condition_type'] })
                  }
                >
                  <option value="region_empty">region_empty</option>
                  <option value="object_present">object_present</option>
                  <option value="person_present">person_present</option>
                  <option value="visual_condition">visual_condition (custom)</option>
                </select>
              </label>
              {previewBlocked && (
                <div className="banner warn small">
                  A custom visual condition must pass a positive/negative
                  preview before it can be activated. Confirm the crop above
                  shows a clear example, then tick below.
                  <label className="row" style={{ marginTop: 8 }}>
                    <input
                      type="checkbox"
                      checked={draft.preview_validated}
                      onChange={(e) => update({ preview_validated: e.target.checked })}
                      style={{ width: 'auto' }}
                    />
                    <span>I checked a positive and a negative example</span>
                  </label>
                </div>
              )}
            </div>

            <div className="card">
              <h2>4 · Temporal confirmation</h2>
              <div className="row">
                <label className="field" style={{ flex: 1 }}>
                  <span>Duration (s)</span>
                  <input
                    type="number"
                    min={1}
                    value={draft.duration_seconds}
                    onChange={(e) =>
                      update({ duration_seconds: Number(e.target.value) })
                    }
                  />
                </label>
                <label className="field" style={{ flex: 1 }}>
                  <span>Min samples</span>
                  <input
                    type="number"
                    min={1}
                    value={draft.min_samples}
                    onChange={(e) => update({ min_samples: Number(e.target.value) })}
                  />
                </label>
              </div>
              <div className="row">
                <label className="field" style={{ flex: 1 }}>
                  <span>Max gap (s)</span>
                  <input
                    type="number"
                    min={1}
                    value={draft.max_gap_seconds}
                    onChange={(e) =>
                      update({ max_gap_seconds: Number(e.target.value) })
                    }
                  />
                </label>
                <label className="field" style={{ flex: 1 }}>
                  <span>Clear samples</span>
                  <input
                    type="number"
                    min={1}
                    value={draft.clear_samples}
                    onChange={(e) => update({ clear_samples: Number(e.target.value) })}
                  />
                </label>
              </div>
              <div className="muted small">
                A condition is confirmed across sampled observations, not proven
                for every instant between frames.
              </div>
            </div>

            <div className="card">
              <h2>5 · Software action</h2>
              <label className="field">
                <span>Action</span>
                <select
                  value={draft.action.type}
                  onChange={(e) =>
                    update({
                      action:
                        e.target.value === 'create_task'
                          ? { type: 'create_task', title: draft.action.title }
                          : {
                              type: 'webhook',
                              destination_id: destinations[0] ?? 'test_receiver',
                              title: draft.action.title,
                            },
                    })
                  }
                >
                  <option value="create_task">Create an internal task</option>
                  <option value="webhook">Deliver a webhook</option>
                </select>
              </label>
              <label className="field">
                <span>Title</span>
                <input
                  type="text"
                  value={draft.action.title}
                  onChange={(e) =>
                    update({ action: { ...draft.action, title: e.target.value } })
                  }
                />
              </label>
              {draft.action.type === 'webhook' && (
                <label className="field">
                  <span>Destination (allowlisted on the backend)</span>
                  <select
                    value={draft.action.destination_id}
                    onChange={(e) =>
                      update({
                        action: { ...draft.action, destination_id: e.target.value },
                      } as Partial<Workflow>)
                    }
                  >
                    {destinations.map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>

            <div className="card">
              <h2>Review</h2>
              <p className="small">{result?.preview_text}</p>
              <div className="row">
                <button
                  className="primary"
                  onClick={() => activate(true)}
                  disabled={busy || previewBlocked}
                >
                  Activate
                </button>
                <button className="ghost" onClick={() => activate(false)} disabled={busy}>
                  Save paused
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="card">
            <div className="empty">
              Describe a task and choose <strong>Review workflow</strong>. The
              compiled rule appears here as editable cards before anything is
              activated.
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
