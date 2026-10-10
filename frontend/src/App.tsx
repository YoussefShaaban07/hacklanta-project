import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { usePolling } from './hooks'
import { Events } from './screens/Events'
import { Monitor } from './screens/Monitor'
import { RuleSetup } from './screens/RuleSetup'
import { Tasks } from './screens/Tasks'
import type { Workflow } from './types'

type Tab = 'monitor' | 'rules' | 'events' | 'tasks'

const TABS: Array<{ id: Tab; label: string }> = [
  { id: 'monitor', label: 'Monitor' },
  { id: 'rules', label: 'Rule setup' },
  { id: 'events', label: 'Events' },
  { id: 'tasks', label: 'Tasks' },
]

export function App() {
  const [tab, setTab] = useState<Tab>('monitor')
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const { data: health } = usePolling(api.health, 5000)

  const loadWorkflows = useCallback(async () => {
    const list = await api.listWorkflows()
    setWorkflows(list)
    setSelectedId((current) => current ?? list[0]?.id ?? null)
    return list
  }, [])

  useEffect(() => {
    loadWorkflows().catch(() => undefined)
  }, [loadWorkflows])

  const selected = workflows.find((w) => w.id === selectedId) ?? null

  return (
    <div className="app">
      <header className="top">
        <h1>Camera Agent</h1>
        <span className="sub">
          describe a condition · review the rule · act on evidence
        </span>
        <span style={{ marginLeft: 'auto' }} className="row">
          {health && (
            <>
              <span className={`pill ${health.camera.health === 'ok' ? 'ok' : 'bad'}`}>
                <span className="dot" />
                camera {health.camera.health}
              </span>
              <span className={`pill ${health.using_real_model ? 'info' : 'warn'}`}>
                {health.vision_model}
              </span>
            </>
          )}
        </span>
      </header>

      {health && !health.using_real_model && (
        <div className="banner warn">
          <strong>Heuristic vision provider.</strong> No <code>GEMINI_API_KEY</code>{' '}
          is configured, so conditions are judged from image statistics (edge
          density and brightness), not visual reasoning. The pipeline is real;
          the perception is a stand-in. Set a key and restart for real model
          output.
        </div>
      )}

      <nav className="tabs">
        {TABS.map((item) => (
          <button
            key={item.id}
            className={tab === item.id ? 'active' : ''}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
        {workflows.length > 1 && (
          <select
            style={{ width: 'auto', marginLeft: 'auto' }}
            value={selectedId ?? ''}
            onChange={(e) => setSelectedId(e.target.value)}
          >
            {workflows.map((w) => (
              <option key={w.id} value={w.id}>
                {w.target_description.slice(0, 48)} (v{w.version})
              </option>
            ))}
          </select>
        )}
      </nav>

      {tab === 'monitor' && (
        <Monitor workflow={selected} onChanged={loadWorkflows} />
      )}
      {tab === 'rules' && (
        <RuleSetup
          workflow={selected}
          destinations={health?.webhook_destinations ?? []}
          onSaved={async (id) => {
            await loadWorkflows()
            setSelectedId(id)
            setTab('monitor')
          }}
          onChanged={loadWorkflows}
        />
      )}
      {tab === 'events' && <Events workflowId={selectedId} />}
      {tab === 'tasks' && <Tasks />}
    </div>
  )
}
