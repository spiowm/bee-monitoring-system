import { useState } from 'react';
import { useMutation, useQuery, useQueries } from '@tanstack/react-query';
import { Target, Loader2, AlertCircle, Sparkles, Info, ChevronDown, ChevronUp, FileDown } from 'lucide-react';
import {
  listEvaluationPairsJobsEvaluateTestPairsGet,
  createEvalJobJobsEvaluatePost,
  getJobJobsJobIdGet,
} from '../api/generated';
import type { TestPair } from '../api/generated';
import type { ProcessConfig, VizConfig, BehaviorEvalResult } from '../types';
import SideBySideVideoPlayer from '../components/SideBySideVideoPlayer';
import BehaviorEvalResults from '../components/BehaviorEvalResults';
import { useLocalStorageState } from '../hooks/useLocalStorageState';

const DEFAULT_CONFIG: ProcessConfig = {
  tracker_name: 'bytetrack',
  approach: 'B',
  line_position: 0.0,
  conf_threshold: 0.2,
  iou_threshold: 0.8,
  max_detections: 1000,
  imgsz: null,
  half_precision: false,
  batch_size: null,
  kp_conf_threshold: 0.5,
  track_tail_length: 30,
  angle_threshold_deg: 60,
  ramp_detect_interval: 30,
  model_name: null,
  // Оптимальні параметри класифікації (відкалібровано, дзеркало ProcessConfig)
  behavior_foraging_speed_min: 100,
  behavior_fanning_max_disp: 60,
  behavior_fanning_duration_min: 0.6,
  behavior_fanning_require_body: false,
  behavior_fanning_priority: true,
  defense_min_appearances: 2,
  stitch_max_dist: 30,
  stitch_max_frames: 45,
};

const DEFAULT_VIZ: VizConfig = {
  show_boxes: true, show_ids: true, show_confidence: false, show_keypoints: false,
  show_ramp: true, show_behaviors: true, show_counting_line: false,
  show_stats_overlay: true, show_tracks: false, show_orientation: true,
  show_recent_events: false,
};

interface JobDoc {
  job_id: string;
  status: string;
  progress: number;
  result?: { annotated_video_url?: string; fps_processed?: number };
  evaluation?: BehaviorEvalResult & { eval_mode?: string };
  error?: string;
}

/* ---------- GT Info Panel ---------- */
function GtInfoPanel() {
  const [open, setOpen] = useState(false);
  return (
    <div className="card border border-blue-900/40">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-2 text-sm font-semibold text-blue-400">
          <Info size={16} /> Що містить Ground Truth датасет?
        </div>
        {open ? <ChevronUp size={16} className="text-gray-500" /> : <ChevronDown size={16} className="text-gray-500" />}
      </button>
      {open && (
        <div className="mt-4 space-y-3 text-sm text-gray-300">
          <p>
            GT-датасет <strong className="text-gray-100">tracking_and_behavior</strong> містить ручні анотації
            з наукової публікації (PLOS ONE). Для кожного відео задано:
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-700 text-gray-400">
                  <th className="text-left py-1.5 pr-4">Компонент</th>
                  <th className="text-center py-1.5 px-2">Є</th>
                  <th className="text-left py-1.5 pl-2">Деталі</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {[
                  ['Bounding boxes', '✅', 'CXCYWH (нормалізовані 0–1), кожен кадр'],
                  ['Track IDs', '✅', 'Унікальний ID бджоли крізь кадри'],
                  ['Поведінка (behavior)', '✅', 'arrival, defensive, fanning, washboarding — бінарні мітки'],
                  ['Entrance zone', '✅', 'Полігон із 4 вершин (пікселі)'],
                  ['Pose keypoints', '❌', 'Голова/жало — відсутні в GT'],
                  ['Рампа (ramp detector)', '❌', 'Визначається нашою моделлю'],
                ].map(([name, status, detail]) => (
                  <tr key={name as string}>
                    <td className="py-1.5 pr-4 font-medium text-gray-200">{name}</td>
                    <td className="py-1.5 px-2 text-center">{status}</td>
                    <td className="py-1.5 pl-2 text-gray-400">{detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="bg-blue-950/30 border border-blue-900/30 rounded-lg p-3 text-xs text-blue-300">
            <strong>Методологія:</strong> fanning/washboarding/defense оцінюються per-frame (IoU-матчинг).
            Foraging оцінюється подієво — перевіряємо чи зафіксував BuzzTrack counting-event для відповідного треку.
            Перші <strong>80 кадрів</strong> кожного треку виключаються з fanning-оцінки (warm-up моделі).
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- Main Page ---------- */
export default function EvaluationPage() {
  const [selectedBasename, setSelectedBasename] = useLocalStorageState<string>('eval_basename', '20230711a-fan_5s');
  const [skipVideo, setSkipVideo] = useLocalStorageState<boolean>('eval_skip_video', false);
  const [jobId, setJobId] = useState<string | null>(null);

  const isRunning = jobId !== null;

  const { data: pairs = [] } = useQuery({
    queryKey: ['eval-pairs'],
    queryFn: async () => {
      const { data } = await listEvaluationPairsJobsEvaluateTestPairsGet();
      return (data as TestPair[]) || [];
    },
  });

  const selectedPair = pairs.find(p => p.basename === selectedBasename);

  const [jobQuery] = useQueries({
    queries: [{
      queryKey: ['eval-job', jobId],
      queryFn: async () => {
        const { data } = await getJobJobsJobIdGet({ path: { job_id: jobId! } });
        return data as JobDoc;
      },
      enabled: !!jobId,
      refetchInterval: (q: { state: { data?: JobDoc } }) => {
        const d = q.state.data;
        if (!jobId) return false;
        if (d?.status === 'failed') return false;
        if (d?.evaluation) return false;
        return 3000;
      },
    }],
  });

  const job = jobQuery.data;
  const isComplete = !!job?.evaluation;

  const runMut = useMutation({
    mutationFn: async () => {
      if (!selectedPair) throw new Error('Не вибрана пара');
      const resp = await createEvalJobJobsEvaluatePost({
        body: {
          filename: selectedPair.video_filename,
          gt_basename: selectedPair.basename,
          config: DEFAULT_CONFIG,
          viz_config: DEFAULT_VIZ,
          eval_mode: 'behavior',
          skip_video: skipVideo,
        },
      });
      return (resp.data as { job_id: string }).job_id;
    },
    onSuccess: (id) => setJobId(id),
  });

  const reset = () => setJobId(null);

  const handleDownloadMd = () => {
    let url = `${import.meta.env.VITE_API_URL || window.location.origin}/analytics/export/md?`;
    if (job) url += `job_a_id=${job.job_id}`;
    window.open(url, '_blank');
  };

  const videos = isComplete && job
    ? [
        ...(job.evaluation?.gt_video_url ? [{
          url: job.evaluation.gt_video_url,
          label: 'Ground Truth',
          borderClass: 'border-yellow-700/60',
          textClass: 'text-yellow-300',
        }] : []),
        ...(job.result?.annotated_video_url ? [{
          url: job.result.annotated_video_url,
          label: 'BuzzTrack',
          borderClass: 'border-blue-700/60',
          textClass: 'text-blue-400',
        }] : []),
      ]
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <h1 className="text-xl font-bold flex items-center gap-2 mb-2">
          <Target size={20} className="text-[var(--accent)]" />
          Оцінка точності поведінки
        </h1>
        <p className="text-sm text-gray-400">
          Порівняння передбачених станів бджіл з ground-truth анотаціями (tracking_and_behavior).
        </p>
      </div>

      {/* GT Info */}
      <GtInfoPanel />

      {/* Controls */}
      <div className="card space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-end">
          {/* Pair selector */}
          <div>
            <label className="block text-xs uppercase font-semibold text-gray-400 mb-1">
              Тестова пара (відео + GT)
            </label>
            <select
              id="eval-pair-select"
              value={selectedBasename}
              onChange={e => setSelectedBasename(e.target.value)}
              disabled={isRunning}
              className="w-full bg-[var(--bg-panel)] border border-gray-700 rounded-lg px-3 py-2 text-sm disabled:opacity-50"
            >
              {pairs.map(p => (
                <option key={p.basename} value={p.basename}>
                  {p.basename} — {p.size_mb} MB
                </option>
              ))}
            </select>
          </div>

          {/* Run button */}
          <button
            id="eval-run-btn"
            onClick={() => runMut.mutate()}
            disabled={isRunning || !selectedPair}
            className="px-6 py-3 bg-[var(--accent)] text-black font-semibold rounded-lg hover:brightness-110 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
          >
            {isRunning ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
            {isRunning ? 'Обробка…' : 'Запустити оцінку'}
          </button>
        </div>

        {/* Skip Video Toggle */}
        <div className="flex items-center gap-2 pt-4 border-t border-gray-800">
          <input
            type="checkbox"
            id="skipVideoCheck"
            checked={skipVideo}
            onChange={(e) => setSkipVideo(e.target.checked)}
            disabled={isRunning}
            className="w-4 h-4 rounded border-gray-700 bg-[var(--bg-panel)] accent-[var(--accent)] cursor-pointer"
          />
          <label htmlFor="skipVideoCheck" className="text-sm text-gray-300 cursor-pointer flex items-center gap-2">
            Без відео виводу (Швидкий тест)
            <span className="text-xs text-gray-500 hidden sm:inline">— пропускає малювання рамок та рендер MP4 (у 2-3x рази швидше)</span>
          </label>
        </div>
      </div>

      {/* Progress */}
      {isRunning && (
        <div className="card space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-[var(--accent)]">Аналіз поведінки…</span>
            <span className="font-mono text-gray-400 text-xs">
              {job?.evaluation
                ? 'готово ✓'
                : job?.status === 'complete'
                  ? 'обчислюємо метрики…'
                  : job?.status || 'очікує…'}
            </span>
          </div>
          <div className="w-full bg-[var(--bg-panel)] rounded-full h-2 overflow-hidden">
            <div
              className="h-full transition-all duration-500 bg-[var(--accent)]"
              style={{ width: `${((job?.progress ?? 0) * 100).toFixed(1)}%` }}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {job?.error && (
        <div className="card border border-red-500/40 flex items-center gap-3">
          <AlertCircle className="text-red-400" />
          <div className="text-sm text-red-300">{job.error}</div>
        </div>
      )}

      {/* Results */}
      {isComplete && job?.evaluation && (
        <div className="space-y-6">
          <div className="card border border-[var(--accent)]/40">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-lg font-bold text-[var(--accent)] flex items-center gap-2">
                <Sparkles size={18} /> Результати оцінки поведінки
              </h2>
              <div className="flex gap-4">
                <button onClick={handleDownloadMd} className="text-xs text-gray-400 hover:text-[var(--accent)] transition flex items-center gap-1">
                  <FileDown size={14} /> Скачати звіт (MD)
                </button>
                <button onClick={reset} className="text-xs text-gray-400 hover:text-gray-200 transition">
                  Нова оцінка
                </button>
              </div>
            </div>
            <div className="text-xs text-gray-500">
              {selectedBasename} · {job.evaluation.fps?.toFixed(0)} fps · warm-up {job.evaluation.warmup_frames ?? 80} кадрів
            </div>
          </div>

          <BehaviorEvalResults result={job.evaluation} />

          {videos.length > 0 && (
            <div className="card space-y-2">
              <div className="text-xs font-semibold text-gray-300 uppercase tracking-wide">
                Ground Truth vs BuzzTrack
              </div>
              <SideBySideVideoPlayer videos={videos} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
