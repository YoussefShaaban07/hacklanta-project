// Mirrors backend/app/schemas.py. Kept in one file so a contract change is a
// single, obvious edit rather than a hunt through components.

export type ConditionType =
  | 'region_empty'
  | 'object_present'
  | 'person_present'
  | 'visual_condition'

export type ObservationResult = 'met' | 'not_met' | 'unknown'
export type EpisodeState = 'IDLE' | 'CANDIDATE' | 'ACTIVE'
export type CameraHealth = 'ok' | 'stale' | 'disconnected'
export type ActionStatus = 'pending' | 'succeeded' | 'failed'

/** Normalized 0..1 against the full frame, never pixels. */
export interface ROI {
  x: number
  y: number
  w: number
  h: number
}

export interface CreateTaskAction {
  type: 'create_task'
  title: string
}

export interface WebhookAction {
  type: 'webhook'
  destination_id: string
  title: string
}

export type WorkflowAction = CreateTaskAction | WebhookAction

export interface Workflow {
  id: string
  version: number
  camera_id: string
  condition_type: ConditionType
  target_description: string
  roi: ROI
  duration_seconds: number
  min_samples: number
  max_gap_seconds: number
  clear_samples: number
  action: WorkflowAction
  enabled: boolean
  reference_image_path: string | null
  preview_validated: boolean
  preview_text?: string
}

export interface Observation {
  id: string
  workflow_id: string
  result: ObservationResult
  explanation: string
  model_id: string
  inference_ms: number
  captured_at_utc: string
  processed_at_utc: string
}

export interface ActionRecord {
  id: string
  event_id: string
  type: 'create_task' | 'webhook'
  status: ActionStatus
  attempts: number
  last_error: string | null
  receipt: string | null
  idempotency_key: string
}

export interface CameraEvent {
  id: string
  workflow_id: string
  workflow_version: number
  episode_id: string
  first_positive_at: string
  triggered_at: string
  status: 'active' | 'unverified' | 'closed'
  snapshot_path: string | null
  clip_path: string | null
  explanation: string
  actions: ActionRecord[]
}

export interface Task {
  id: string
  title: string
  workflow_id: string
  event_id: string
  created_at: string
  completed: number
  evidence_path: string | null
}

/** The single label the Monitor screen shows for the inference path. */
export type DisplayState =
  | 'monitoring'
  | 'confirming'
  | 'event active'
  | 'unverified'
  | 'uncertain'
  | 'disconnected'
  | 'stale'
  | 'paused'
  | 'starting'
  | 'not running'

export interface WorkflowStatus {
  workflow: Workflow
  preview_text: string
  camera: CameraStatus
  worker_running: boolean
  samples_taken: number
  episode_state: EpisodeState
  condition_unverified: boolean
  positive_samples: number
  clear_streak: number
  camera_health: CameraHealth
  latest_observation: Observation | null
  buffered_frames?: number
  pending_clips?: number
  display_state: DisplayState
}

export interface CameraStatus {
  source: string
  health: CameraHealth
  frame_id: number | null
  last_frame_at: string | null
  age_seconds: number | null
}

export interface Camera extends CameraStatus {
  id: string
  source_kind: string
  is_live: boolean
}

export interface Health {
  status: string
  camera: CameraStatus
  vision_provider: string
  vision_model: string
  using_real_model: boolean
  active_workflows: number
  sample_interval_seconds: number
  webhook_destinations: string[]
}

export interface CompileResponse {
  ok: boolean
  draft: Workflow | null
  questions: string[]
  unsupported_reason: string | null
  preview_text: string
  requires_preview: boolean
}

export interface HistoryAnswer {
  supported: boolean
  kind: string
  answer: string
  events: CameraEvent[]
  actions: ActionRecord[]
  scope_note: string
  suggestions: string[]
}
