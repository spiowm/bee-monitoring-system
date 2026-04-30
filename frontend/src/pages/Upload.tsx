import { useState, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
  createJobJobsPost,
  createTestJobJobsTestPost,
  getJobJobsJobIdGet,
  getJobLiveStatsJobsJobIdLiveGet,
  listTestVideosJobsTestVideosGet,
  deleteJobJobsJobIdDelete,
} from '../api/generated';
import type { ProcessConfig, VizConfig, Job, LiveStats } from '../types';
import JobConfigPanel from '../components/JobConfigPanel';
import LiveStatsPanel from '../components/LiveStatsPanel';
import { Video, Download, AlertCircle, Square } from 'lucide-react';

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const [config, setConfig] = useState({
    tracker_name: 'bytetrack', approach: 'A', line_position: 0.5,
    conf_threshold: 0.20, kp_conf_threshold: 0.5, track_tail_length: 30,
    angle_threshold_deg: 60.0, ramp_detect_interval: 30,
    model_name: null as string | null,
    behavior_foraging_speed_min: 100,
    behavior_fanning_speed_max: 15,
    behavior_fanning_duration_min: 2.0,
    behavior_guarding_speed_min: 15,
    behavior_guarding_speed_max: 80,
    behavior_guarding_spread_ratio: 1.5,
  } as unknown as ProcessConfig);

  const [vizConfig, setVizConfig] = useState<VizConfig>({
    show_boxes: true, show_ids: true, show_confidence: true,
    show_keypoints: true, show_ramp: true, show_behaviors: true,
    show_counting_line: true, show_stats_overlay: true, show_tracks: true,
    show_orientation: true, show_recent_events: true
  });

  const { data: testVideos = [] } = useQuery({
    queryKey: ['testVideos'],
    queryFn: async () => {
      const { data } = await listTestVideosJobsTestVideosGet();
      return (data as string[]) || [];
    }
  });

  const { data: liveStatsData } = useQuery({
    queryKey: ['liveStats', jobId],
    queryFn: async () => {
      const { data } = await getJobLiveStatsJobsJobIdLiveGet({ path: { job_id: jobId! } });
      return data as { live_stats: LiveStats, status: string, progress: number };
    },
    enabled: !!jobId && isProcessing,
    refetchInterval: isProcessing ? 2000 : false,
  });

  const liveStats = liveStatsData?.live_stats || null;
  
  useEffect(() => {
    if (liveStatsData?.status === 'complete' || liveStatsData?.status === 'failed') {
      setIsProcessing(false);
    }
  }, [liveStatsData?.status]);

  const { data: jobData } = useQuery({
    queryKey: ['job', jobId],
    queryFn: async () => {
      const { data } = await getJobJobsJobIdGet({ path: { job_id: jobId! } });
      return data as Job;
    },
    enabled: !!jobId && !isProcessing && (liveStatsData?.status === 'complete' || liveStatsData?.status === 'failed'),
  });

  const job = jobData || null;

  const createJobMut = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("No file");
      const { data } = await createJobJobsPost({
        body: { video: file, config: JSON.stringify(config), viz_config: JSON.stringify(vizConfig) }
      });
      return data;
    },
    onSuccess: (data) => {
      if (data) {
        setJobId(data.job_id);
        setIsProcessing(true);
      }
    }
  });

  const createTestJobMut = useMutation({
    mutationFn: async (filename: string) => {
      const { data } = await createTestJobJobsTestPost({
        body: { filename, config, viz_config: vizConfig }
      });
      return data;
    },
    onSuccess: (data) => {
      if (data) {
        setJobId(data.job_id);
        setIsProcessing(true);
        setFile(null);
      }
    }
  });

  const stopMut = useMutation({
    mutationFn: async (id: string) => {
      await deleteJobJobsJobIdDelete({ path: { job_id: id } });
    },
    onSuccess: () => {
      setJobId(null);
      setIsProcessing(false);
    },
  });

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-full flex-grow">
      {/* 1. LEFT COLUMN: Config */}
      <div className="w-full lg:w-[320px] shrink-0 space-y-4">
        <JobConfigPanel 
          file={file} setFile={setFile}
          config={config} setConfig={setConfig}
          vizConfig={vizConfig} setVizConfig={setVizConfig}
          testVideos={testVideos}
          isProcessing={isProcessing}
          jobId={jobId}
          onStart={() => createJobMut.mutate()}
          onStartTest={(filename) => createTestJobMut.mutate(filename)}
        />
      </div>

      {/* 2. CENTER COLUMN: Video Player */}
      <div className="flex-grow flex flex-col gap-4 min-w-0">
        <div className="card flex-grow flex flex-col items-center justify-center p-0 overflow-hidden relative min-h-[400px]">
          {!jobId && !isProcessing && (
            <div className="text-center text-gray-600 select-none">
              <div className="w-20 h-20 rounded-full border-2 border-dashed border-gray-700 flex items-center justify-center mx-auto mb-4">
                <Video size={36} className="opacity-30" />
              </div>
              <p className="text-sm text-gray-500">Оберіть відео і натисніть</p>
              <p className="text-xs text-gray-700 mt-1">Запустити аналіз</p>
            </div>
          )}

          {isProcessing && (
            <div className="w-full max-w-sm p-8 text-center">
              <div className="w-16 h-16 rounded-full border-4 border-[var(--accent)]/20 border-t-[var(--accent)] animate-spin mx-auto mb-5" />
              <h3 className="text-lg font-bold mb-1 text-gray-100">Аналізується відео</h3>
              <p className="text-xs text-gray-500 mb-4">
                Кадр {liveStats?.current_frame || 0} / {liveStats?.total_frames || '?'}
              </p>
              <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden mb-5">
                <div
                  className="bg-[var(--accent)] h-full rounded-full transition-all duration-300"
                  style={{ width: `${(liveStats?.current_frame || 0) / Math.max(1, liveStats?.total_frames || 1) * 100}%` }}
                />
              </div>
              <button
                onClick={() => jobId && stopMut.mutate(jobId)}
                disabled={stopMut.isPending}
                className="flex items-center gap-2 mx-auto text-sm px-4 py-2 rounded-lg bg-red-900/40 hover:bg-red-800/60 border border-red-700/50 text-red-300 transition-colors disabled:opacity-50"
              >
                <Square size={13} />
                {stopMut.isPending ? 'Зупиняємось…' : 'Стоп і видалити'}
              </button>
            </div>
          )}

          {job && job.status === 'complete' && job.result && (
            <div className="w-full h-full flex flex-col animate-in fade-in zoom-in-95 duration-300">
              <div className="bg-black flex-grow flex items-center justify-center relative">
                 <video src={`${import.meta.env.VITE_API_URL || window.location.origin}${job.result.annotated_video_url}`} controls className="max-h-full max-w-full" />
              </div>
              <div className="p-4 bg-[var(--bg-panel)] flex justify-between items-center border-t border-gray-800">
                <div>
                   <span className="font-bold text-[var(--accent)] text-lg">Аналіз завершено</span>
                   <p className="text-xs text-gray-400">Оброблено за {job.result.duration_sec.toFixed(1)}с ({job.result.fps_processed.toFixed(1)} кадрів/с)</p>
                </div>
                <a href={`${import.meta.env.VITE_API_URL || window.location.origin}${job.result.annotated_video_url}`} download className="flex items-center gap-2 text-sm bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg transition-colors">
                  <Download size={16} /> Завантажити відео
                </a>
              </div>
            </div>
          )}
          
          {job && job.status === 'failed' && (
            <div className="text-red-400 text-center">
              <AlertCircle size={48} className="mx-auto mb-2" />
              <p>Помилка обробки</p>
              <p className="text-xs opacity-70 mt-2">{job.error}</p>
            </div>
          )}
        </div>
      </div>

      {/* 3. RIGHT COLUMN: Live Stats */}
      <div className="w-full xl:w-[320px] shrink-0 space-y-4">
        <LiveStatsPanel liveStats={liveStats} job={job} />
      </div>
      
    </div>
  );
}
