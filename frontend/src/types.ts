// These types bridge the gap between our generated OpenAPI types (which sometimes lack specific return types)
// and our React components.

import type { ProcessConfig as GeneratedProcessConfig, VizConfig as GeneratedVizConfig } from './api/generated';

export type ProcessConfig = GeneratedProcessConfig;
export type VizConfig = GeneratedVizConfig;

export interface Job {
  job_id: string;
  filename: string;
  status: 'pending' | 'processing' | 'complete' | 'failed';
  progress: number;
  created_at: string;
  config: ProcessConfig;
  viz_config: VizConfig;
  live_stats: LiveStats;
  result?: JobResult;
  error?: string;
}

export interface LiveStats {
  current_frame: number;
  total_frames: number;
  bees_on_ramp: number;
  total_in: number;
  total_out: number;
  current_fps: number;
  tracker_name: string;
  approach: string;
  pose_confirmed: number;
  fallback_events: number;
  pose_rejected?: number;
  behavior_counts: Record<string, number>;
  recent_events?: Array<{
    frame: number;
    timestamp_sec: number;
    track_id: number;
    direction: 'IN' | 'OUT';
    method: string;
    speed_px_per_sec: number;
    behavior_class?: string | null;
    angle_deg?: number | null;
  }>;
}

export interface JobResult {
  total_in: number;
  total_out: number;
  total_frames: number;
  duration_sec: number;
  fps_processed: number;
  approach_used: string;
  tracker_used: string;
  pose_confirmed_events: number;
  fallback_events: number;
  pose_rejected_events?: number;
  ramp_detected: boolean;
  annotated_video_url: string;
  behavior_summary: Record<string, number>;
  events: EventRecord[];
  rejected_events?: EventRecord[];
  recommendations?: Recommendation[];
  defense_events?: DefenseEvent[];
}

export interface Recommendation {
  severity: 'info' | 'warning' | 'critical';
  icon: string;
  title: string;
  description: string;
  action: string | null;
}

export interface DefenseEvent {
  frame: number;
  thief_track_id: number;
  defender_track_ids: number[];
  center: [number, number];
  radius: number;
}

export interface EventRecord {
  frame: number;
  timestamp_sec: number;
  track_id: number;
  direction: "IN" | "OUT";
  method: "pose_confirmed" | "trajectory_fallback" | "trajectory_only" | "rejected";
  speed_px_per_sec: number;
  behavior_class: string | null;
  angle_deg: number | null;
}
