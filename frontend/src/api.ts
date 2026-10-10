// Typed API client. Every network call in the app goes through here.

import type {
  Camera,
  CameraEvent,
  CompileResponse,
  Health,
  HistoryAnswer,
  ROI,
  Task,
  Workflow,
  WorkflowStatus,
} from './types'

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* non-JSON error body; keep the status line */
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<Health>('/health'),

  cameras: () => request<Camera[]>('/cameras'),

  /** Cache-busted so the browser cannot show a stale frame as current. */
  snapshotUrl: (cameraId: string) =>
    `${BASE}/cameras/${cameraId}/snapshot?t=${Date.now()}`,

  previewUrl: (cameraId: string) => `${BASE}/cameras/${cameraId}/preview`,

  /** Exactly the crop the model will inspect. */
  cropPreview: async (cameraId: string, roi: ROI): Promise<string> => {
    const response = await fetch(`${BASE}/cameras/${cameraId}/crop-preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(roi),
    })
    if (!response.ok) throw new Error(`crop preview failed: ${response.status}`)
    const blob = await response.blob()
    return URL.createObjectURL(blob)
  },

  compile: (instruction: string, cameraId: string, roi?: ROI) =>
    request<CompileResponse>('/workflows/compile', {
      method: 'POST',
      body: JSON.stringify({ instruction, camera_id: cameraId, roi }),
    }),

  listWorkflows: () => request<Workflow[]>('/workflows'),

  saveWorkflow: (workflow: Workflow) =>
    request<Workflow>('/workflows', {
      method: 'POST',
      body: JSON.stringify({ workflow }),
    }),

  patchWorkflow: (id: string, patch: Partial<Workflow>) =>
    request<Workflow & { version_bumped: boolean }>(`/workflows/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  status: (id: string) => request<WorkflowStatus>(`/workflows/${id}/status`),

  events: (workflowId?: string, limit = 50) =>
    request<CameraEvent[]>(
      `/events?limit=${limit}${workflowId ? `&workflow_id=${workflowId}` : ''}`,
    ),

  event: (id: string) => request<CameraEvent>(`/events/${id}`),

  tasks: () => request<Task[]>('/tasks'),

  retryAction: (actionId: string) =>
    request<unknown>(`/actions/${actionId}/retry`, { method: 'POST' }),

  askHistory: (question: string, workflowId?: string) =>
    request<HistoryAnswer>('/history/query', {
      method: 'POST',
      body: JSON.stringify({ question, workflow_id: workflowId }),
    }),

  /** Evidence is served by filename from the backend's evidence directory. */
  evidenceUrl: (path: string | null) => {
    if (!path) return null
    const name = path.split('/').pop()
    return name ? `${BASE}/evidence/${name}` : null
  },
}
