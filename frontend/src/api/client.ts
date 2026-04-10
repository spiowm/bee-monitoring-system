import axios from 'axios';

export const apiClient = axios.create({
  baseURL: 'http://localhost:8000',
});

export interface ProcessConfig {
  tracker_name: string;
  approach: string;
  line_position: number;
  conf_threshold: number;
  kp_conf_threshold: number;
  track_tail_length: number;
  angle_threshold_deg: number;
  ramp_detect_interval: number;
}

export interface VizConfig {
  show_boxes: boolean;
  show_ids: boolean;
  show_confidence: boolean;
  show_keypoints: boolean;
  show_ramp: boolean;
  show_behaviors: boolean;
  show_counting_line: boolean;
  show_stats_overlay: boolean;
  show_tracks: boolean;
  show_orientation: boolean;
  show_recent_events: boolean;
}

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
  behavior_counts: Record<string, number>;
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
  ramp_detected: boolean;
  annotated_video_url: string;
  behavior_summary: Record<string, number>;
  events: EventRecord[];
}

export interface EventRecord {
  frame: number;
  timestamp_sec: number;
  track_id: number;
  direction: "IN" | "OUT";
  method: "pose_confirmed" | "trajectory_fallback" | "trajectory_only";
  speed_px_per_sec: number;
  behavior_class: string | null;
  angle_deg: number | null;
}

export const API = {
  createJob: async (file: File, config: ProcessConfig, vizConfig: VizConfig) => {
    const formData = new FormData();
    formData.append('video', file);
    formData.append('config', JSON.stringify(config));
    formData.append('viz_config', JSON.stringify(vizConfig));
    const { data } = await apiClient.post<{job_id: string, status: string}>('/jobs', formData);
    return data;
  },
  createTestJob: async (filename: string, config: ProcessConfig, vizConfig: VizConfig) => {
    const { data } = await apiClient.post<{job_id: string, status: string}>('/jobs/test', {
      filename,
      config,
      viz_config: vizConfig
    });
    return data;
  },
  getJob: async (jobId: string) => {
    const { data } = await apiClient.get<Job>(`/jobs/${jobId}`);
    return data;
  },
  deleteJob: async (jobId: string) => {
    const { data } = await apiClient.delete(`/jobs/${jobId}`);
    return data;
  },
  getTestVideos: async () => {
    const { data } = await apiClient.get<string[]>('/jobs/test/videos');
    return data;
  },
  getJobLiveStats: async (jobId: string) => {
    const { data } = await apiClient.get<{live_stats: LiveStats, status: string, progress: number}>(`/jobs/${jobId}/live`);
    return data;
  },
  listJobs: async () => {
    const { data } = await apiClient.get<Job[]>('/jobs');
    return data;
  },
  getAnalyticsSummary: async () => {
    const { data } = await apiClient.get('/analytics/summary');
    return data;
  },
  getApproachCompare: async () => {
    const { data } = await apiClient.get('/analytics/compare-approaches');
    return data;
  }
};
