import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQueries, useQuery } from '@tanstack/react-query';
import {
  AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { Target, Loader2, AlertCircle, Sparkles, Info, ChevronDown, ChevronUp, FileDown } from 'lucide-react';
import {
  listEvaluationPairsJobsEvaluateTestPairsGet,
  createEvalJobJobsEvaluatePost,
  getJobJobsJobIdGet,
} from '../api/generated';
import type { TestPair } from '../api/generated';
import type { ProcessConfig, VizConfig } from '../types';
import EvaluationKPICards from '../components/EvaluationKPICards';
import SideBySideVideoPlayer from '../components/SideBySideVideoPlayer';
import type { VideoSource } from '../components/SideBySideVideoPlayer';
import ApproachWinnerBars from '../components/ApproachWinnerBars';
import DetailedMetricsTable from '../components/DetailedMetricsTable';
import BehaviorEvalResults from '../components/BehaviorEvalResults';
import { useLocalStorageState } from '../hooks/useLocalStorageState';

const DEFAULT_CONFIG: ProcessConfig = {
  tracker_name: 'bytetrack',
  approach: 'A',
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
  behavior_foraging_speed_min: 100,
  behavior_fanning_speed_max: 15,
  behavior_fanning_duration_min: 2,
  behavior_guarding_speed_min: 15,
  behavior_guarding_speed_max: 80,
  behavior_guarding_spread_ratio: 1.5,
};

const DEFAULT_VIZ: VizConfig = {
  show_boxes: true, show_ids: true, show_confidence: false, show_keypoints: false,
  show_ramp: true, show_behaviors: false, show_counting_line: true,
  show_stats_overlay: true, show_tracks: false, show_orientation: true,
  show_recent_events: false,
};

interface GtEvent { frame: number; timestamp_sec: number; track_id: number; direction: string; }
interface PredEvent { frame: number; timestamp_sec: number; direction: string; }
interface DirMetrics {
  tp: number; fp: number; fn: number;
  precision: number; recall: number; f1: number; mae: number;
  gt_count: number; pred_count: number;
}
interface EvaluationDoc {
  in: DirMetrics;
  out: DirMetrics;
  accuracy: number;
  gt_total_in: number; gt_total_out: number;
  pred_total_in: number; pred_total_out: number;
  gt_video_url: string;
  gt_basename: string;
  gt_events: GtEvent[];
  fps: number;
}
interface JobDoc {
  job_id: string;
  status: string;
  progress: number;
  result?: { events?: PredEvent[]; annotated_video_url?: string; fps_processed?: number };
  evaluation?: EvaluationDoc;
  error?: string;
  config?: ProcessConfig;
}

function buildCumulativeData(gt: GtEvent[], predA: PredEvent[], predB: PredEvent[] | null, fps: number) {
  const allFrames = [...gt, ...predA, ...(predB || [])].map(e => e.frame);
  const maxFrame = allFrames.length ? Math.max(...allFrames) : 1;
  const bucket = Math.max(1, Math.floor(maxFrame / 60));
  const numBuckets = Math.ceil(maxFrame / bucket) + 1;

  const data = Array.from({ length: numBuckets }, (_, i) => ({
    t: ((i * bucket) / fps).toFixed(1),
    gt: 0, predA: 0, ...(predB ? { predB: 0 } : {}),
  }));

  const tally = (arr: { frame: number }[], key: string) => {
    let count = 0;
    let cursor = 0;
    const sorted = [...arr].sort((a, b) => a.frame - b.frame);
    for (let i = 0; i < numBuckets; i++) {
      const limit = (i + 1) * bucket;
      while (cursor < sorted.length && sorted[cursor].frame <= limit) {
        count++;
        cursor++;
      }
      (data[i] as Record<string, unknown>)[key] = count;
    }
  };

  tally(gt, 'gt');
  tally(predA, 'predA');
  if (predB) tally(predB, 'predB');
  return data;
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
                  ['Поведінка (behavior)', '✅', 'arrival, defensive, fanning — бінарні мітки'],
                  ['Entrance zone', '✅', 'Полігон із 4 вершин (пікселі)'],
                  ['Pose keypoints', '❌', 'Голова/жало — відсутні в GT'],
                  ['Рампа (ramp detector)', '❌', 'Визначається нашою моделлю'],
                  ['Counting events', '⚙️', 'Обчислюються з GT-треків автоматично'],
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
            <strong>Як порівнюємо:</strong> GT-events (вліт/виліт) обчислюються з GT-треків
            за тією ж логікою лінії підрахунку, що й наша модель. Потім зіставляємо з
            передбаченими подіями (±15 кадрів) → TP, FP, FN, F1, MAE.
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- Main Page ---------- */
export default function EvaluationPage() {
  const [selectedBasename, setSelectedBasename] = useLocalStorageState<string>('eval_basename', '20230711a-fan_5s');
  const [approach, setApproach] = useLocalStorageState<'A' | 'B' | 'AB'>('eval_approach', 'A');
  const [evalMode, setEvalMode] = useLocalStorageState<'counting' | 'behavior'>('eval_mode', 'counting');
  const [skipVideo, setSkipVideo] = useLocalStorageState<boolean>('eval_skip_video', false);
  const [jobIds, setJobIds] = useState<{ a: string | null; b: string | null }>({ a: null, b: null });
  const [pendingJobBBody, setPendingJobBBody] = useState<any>(null);

  const isRunning = jobIds.a !== null || jobIds.b !== null || pendingJobBBody !== null;
  const isBothMode = approach === 'AB';

  const { data: pairs = [] } = useQuery({
    queryKey: ['eval-pairs'],
    queryFn: async () => {
      const { data } = await listEvaluationPairsJobsEvaluateTestPairsGet();
      return (data as TestPair[]) || [];
    },
  });

  const selectedPair = pairs.find(p => p.basename === selectedBasename);

  const jobQueries = useQueries({
    queries: (['a', 'b'] as const).map(slot => ({
      queryKey: ['eval-job', slot, jobIds[slot]],
      queryFn: async () => {
        const { data } = await getJobJobsJobIdGet({ path: { job_id: jobIds[slot]! } });
        return data as JobDoc;
      },
      enabled: !!jobIds[slot],
      refetchInterval: (q: { state: { data?: JobDoc } }) => {
        const data = q.state.data;
        if (!jobIds[slot]) return false;
        if (data?.status === 'failed') return false;
        if (data?.evaluation) return false;
        return 3000;
      },
    })),
  });

  const jobA = jobQueries[0].data;
  const jobB = jobQueries[1].data;

  // Визначаємо завершеність залежно від режиму
  const singleComplete = !isBothMode && !!jobA?.evaluation;
  const bothComplete = isBothMode && !!(jobA?.evaluation && jobB?.evaluation);
  const isComplete = singleComplete || bothComplete;

  const runMut = useMutation({
    mutationFn: async () => {
      if (!selectedPair) throw new Error('Не вибрана пара');
      const baseBody = {
        filename: selectedPair.video_filename,
        gt_basename: selectedPair.basename,
        viz_config: DEFAULT_VIZ,
        eval_mode: evalMode,
        skip_video: skipVideo,
      };

      if (isBothMode) {
        const respA = await createEvalJobJobsEvaluatePost({
          body: { ...baseBody, config: { ...DEFAULT_CONFIG, approach: 'A' } },
        });
        
        // Keep config for Job B to be launched sequentially
        setPendingJobBBody({ ...baseBody, config: { ...DEFAULT_CONFIG, approach: 'B' } });
        
        return {
          jobIdA: (respA.data as { job_id: string }).job_id,
          jobIdB: null,
        };
      } else {
        const resp = await createEvalJobJobsEvaluatePost({
          body: { ...baseBody, config: { ...DEFAULT_CONFIG, approach } },
        });
        return {
          jobIdA: (resp.data as { job_id: string }).job_id,
          jobIdB: null,
        };
      }
    },
    onSuccess: ({ jobIdA, jobIdB }) => {
      setJobIds({ a: jobIdA, b: jobIdB });
    },
  });

  useEffect(() => {
    // Launch Job B sequentially after Job A completes
    if (pendingJobBBody && jobA?.status === 'complete' && jobA?.evaluation && !jobIds.b) {
      const runB = async () => {
        try {
          const respB = await createEvalJobJobsEvaluatePost({ body: pendingJobBBody });
          setJobIds(prev => ({ ...prev, b: (respB.data as { job_id: string }).job_id }));
          setPendingJobBBody(null);
        } catch (e) {
          console.error("Failed to start job B sequentially", e);
        }
      };
      runB();
    }
  }, [pendingJobBBody, jobA?.status, jobA?.evaluation, jobIds.b]);

  const reset = () => {
    setJobIds({ a: null, b: null });
    setPendingJobBBody(null);
  };

  const handleDownloadMd = () => {
    let url = `${import.meta.env.VITE_API_URL || window.location.origin}/analytics/export/md?`;
    if (jobA) url += `job_a_id=${jobA.job_id}&`;
    if (jobB) url += `job_b_id=${jobB.job_id}`;
    
    window.open(url, '_blank');
  };

  const cumulativeData = useMemo(() => {
    if (!isComplete || !jobA?.evaluation) return [];
    return buildCumulativeData(
      jobA.evaluation.gt_events,
      jobA.result?.events || [],
      isBothMode && jobB?.evaluation ? (jobB.result?.events || []) : null,
      jobA.evaluation.fps || 30,
    );
  }, [isComplete, jobA, jobB, isBothMode]);

  const videos: VideoSource[] = useMemo(() => {
    if (!isComplete || !jobA) return [];
    const result: VideoSource[] = [];

    const labelA = approach === 'B' ? 'Метод Б — pose-validated' : 'Метод А — траєкторія';
    result.push({
      url: jobA.result!.annotated_video_url!,
      label: isBothMode ? 'Метод А — траєкторія' : labelA,
      sublabel: `${jobA.evaluation!.pred_total_in} IN / ${jobA.evaluation!.pred_total_out} OUT`,
      borderClass: approach === 'B' && !isBothMode ? 'border-[var(--accent)]/60' : 'border-blue-700/60',
      textClass: approach === 'B' && !isBothMode ? 'text-[var(--accent)]' : 'text-blue-400',
    });

    if (isBothMode && jobB?.evaluation) {
      result.push({
        url: jobB.result!.annotated_video_url!,
        label: 'Метод Б — pose-validated',
        sublabel: `${jobB.evaluation!.pred_total_in} IN / ${jobB.evaluation!.pred_total_out} OUT`,
        borderClass: 'border-[var(--accent)]/60',
        textClass: 'text-[var(--accent)]',
      });
    }

    result.push({
      url: jobA.evaluation!.gt_video_url,
      label: 'Ground Truth',
      sublabel: `${jobA.evaluation!.gt_total_in} IN / ${jobA.evaluation!.gt_total_out} OUT`,
      borderClass: 'border-yellow-700/60',
      textClass: 'text-yellow-300',
    });

    return result;
  }, [isComplete, jobA, jobB, approach, isBothMode]);

  const approachLabel = approach === 'A' ? 'А (траєкторія)' : approach === 'B' ? 'Б (pose)' : 'А ↔ Б';
  const activeSlots = isBothMode
    ? [
        { slot: 'a' as const, label: 'Метод А — траєкторія', color: '#3b82f6' },
        { slot: 'b' as const, label: 'Метод Б — pose-validated', color: '#f0b429' },
      ]
    : [{ slot: 'a' as const, label: `Метод ${approachLabel}`, color: approach === 'B' ? '#f0b429' : '#3b82f6' }];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <h1 className="text-xl font-bold flex items-center gap-2 mb-2">
          <Target size={20} className="text-[var(--accent)]" />
          Оцінка точності системи
        </h1>
        <p className="text-sm text-gray-400">
          Порівняння результатів детекції з ground-truth анотаціями (tracking_and_behavior).
          GT-events обчислюються з ручних треків за тією ж лінією підрахунку.
        </p>
      </div>

      {/* GT Info */}
      <GtInfoPanel />

      {/* Controls */}
      <div className="card space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
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

          {/* Approach selector */}
          {evalMode === 'counting' && (
            <div>
              <label className="block text-xs uppercase font-semibold text-gray-400 mb-1">
                Метод
              </label>
              <div className="flex gap-1">
                {([
                  { val: 'A' as const, label: 'А (траєкторія)' },
                  { val: 'B' as const, label: 'Б (pose)' },
                  { val: 'AB' as const, label: 'А ↔ Б' },
                ]).map(({ val, label }) => (
                  <button
                    key={val}
                    onClick={() => setApproach(val)}
                    disabled={isRunning}
                    className={`flex-1 px-3 py-2 text-xs font-medium rounded-lg border transition ${
                      approach === val
                        ? 'bg-[var(--accent)]/20 border-[var(--accent)]/60 text-[var(--accent)]'
                        : 'bg-[var(--bg-panel)] border-gray-700 text-gray-400 hover:border-gray-500'
                    } disabled:opacity-50`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Eval Mode selector */}
          <div>
            <label className="block text-xs uppercase font-semibold text-gray-400 mb-1">
              Режим
            </label>
            <select
              value={evalMode}
              onChange={e => setEvalMode(e.target.value as 'counting' | 'behavior')}
              disabled={isRunning}
              className="w-full bg-[var(--bg-panel)] border border-gray-700 rounded-lg px-3 py-2 text-sm disabled:opacity-50"
            >
              <option value="counting">Підрахунок (IN/OUT)</option>
              <option value="behavior">Точність станів</option>
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
            {isRunning ? 'Обробка…' : `Запустити ${approachLabel}`}
          </button>
        </div>
        
        {/* Skip Video Toggle */}
        <div className="flex items-center gap-2 mt-4 pt-4 border-t border-gray-800">
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

      {/* Progress bars */}
      {isRunning && (
        <div className={`grid grid-cols-1 ${isBothMode ? 'md:grid-cols-2' : ''} gap-4`}>
          {activeSlots.map(({ slot, label, color }) => {
            const job = slot === 'a' ? jobA : jobB;
            const progress = job?.progress ?? 0;
            const status = job?.evaluation
              ? 'готово ✓'
              : job?.status === 'complete'
                ? 'обчислюємо метрики…'
                : job?.status || 'очікує…';
            return (
              <div key={slot} className="card space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span style={{ color }}>{label}</span>
                  <span className="font-mono text-gray-400 text-xs">{status}</span>
                </div>
                <div className="w-full bg-[var(--bg-panel)] rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full transition-all duration-500"
                    style={{ width: `${(progress * 100).toFixed(1)}%`, background: color }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Errors */}
      {(jobA?.error || jobB?.error) && (
        <div className="card border border-red-500/40 flex items-center gap-3">
          <AlertCircle className="text-red-400" />
          <div className="text-sm">
            {jobA?.error && <div>Метод А: {jobA.error}</div>}
            {jobB?.error && <div>Метод Б: {jobB.error}</div>}
          </div>
        </div>
      )}

      {/* Results: Single approach */}
      {singleComplete && jobA?.evaluation && (
        <>
          <div className="card border border-[var(--accent)]/40">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-[var(--accent)] flex items-center gap-2">
                <Sparkles size={18} /> Результати — Метод {approach === 'B' ? 'Б' : 'А'}
              </h2>
              <div className="flex gap-4">
                <button onClick={handleDownloadMd} className="text-xs text-gray-400 hover:text-[var(--accent)] transition flex items-center gap-1">
                  <FileDown size={14} /> Скачати звіт (MD)
                </button>
                <button onClick={reset} className="text-xs text-gray-400 hover:text-gray-200 transition">
                  Нове порівняння
                </button>
              </div>
            </div>
            <EvaluationKPICards
              accuracy={jobA.evaluation.accuracy}
              in_metrics={jobA.evaluation.in}
              out_metrics={jobA.evaluation.out}
            />
          </div>

          <DetailedMetricsTable 
            a={{ ...jobA.evaluation, fps: jobA.result?.fps_processed }}
            labelA={approach === 'B' ? 'Метод Б (pose)' : 'Метод А (траєкторія)'}
          />

          <div className="card space-y-2">
            <div className="text-xs font-semibold text-gray-300 uppercase tracking-wide">
              Відео: Модель vs Ground Truth
            </div>
            {videos.length > 0 && <SideBySideVideoPlayer videos={videos} />}
          </div>

          <div className="card">
            <div className="text-xs font-semibold mb-2 text-gray-300 uppercase tracking-wide">
              Кумулятивні події в часі
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={cumulativeData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="t" tick={{ fontSize: 11, fill: '#9ca3af' }} label={{ value: 'sec', offset: -2, position: 'insideBottom', fill: '#9ca3af', fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} />
                <Tooltip contentStyle={{ background: '#161922', border: '1px solid #374151' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Area type="monotone" dataKey="gt" name="Ground Truth" stroke="#facc15" fill="#facc1530" />
                <Area type="monotone" dataKey="predA" name={`Метод ${approach === 'B' ? 'Б' : 'А'}`} stroke={approach === 'B' ? '#f0b429' : '#3b82f6'} fill="transparent" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* Results: Both approaches */}
      {bothComplete && jobA?.evaluation && jobB?.evaluation && (
        <>
          <div className="card border-2 border-[var(--accent)] shadow-[0_0_30px_rgba(240,180,41,0.15)] bg-gradient-to-br from-[var(--bg-card)] to-[var(--bg-panel)]">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-[var(--accent)] uppercase tracking-wider flex items-center gap-2">
                <Sparkles size={20} /> Порівняння методів
              </h2>
              <div className="flex gap-4">
                <button onClick={handleDownloadMd} className="text-xs text-gray-400 hover:text-[var(--accent)] transition flex items-center gap-1">
                  <FileDown size={14} /> Скачати звіт (MD)
                </button>
                <button onClick={reset} className="text-xs text-gray-400 hover:text-gray-200 transition">
                  Нове порівняння
                </button>
              </div>
            </div>
            <ApproachWinnerBars a={jobA.evaluation} b={jobB.evaluation} />
          </div>

          <div className="space-y-3">
            <div className="text-sm font-semibold text-blue-400 uppercase tracking-wider">Метод А — деталі</div>
            <EvaluationKPICards accuracy={jobA.evaluation.accuracy} in_metrics={jobA.evaluation.in} out_metrics={jobA.evaluation.out} />
          </div>
          <div className="space-y-3">
            <div className="text-sm font-semibold text-[var(--accent)] uppercase tracking-wider">Метод Б — деталі</div>
            <EvaluationKPICards accuracy={jobB.evaluation.accuracy} in_metrics={jobB.evaluation.in} out_metrics={jobB.evaluation.out} />
          </div>

          <DetailedMetricsTable 
            a={{ ...jobA.evaluation, fps: jobA.result?.fps_processed }}
            b={{ ...jobB.evaluation, fps: jobB.result?.fps_processed }}
            labelA="Метод А (траєкторія)"
            labelB="Метод Б (pose-validated)"
          />

          <div className="card space-y-2">
            <div className="text-xs font-semibold text-gray-300 uppercase tracking-wide">
              Side-by-side: Метод А · Метод Б · Ground Truth
            </div>
            {videos.length > 0 && <SideBySideVideoPlayer videos={videos} />}
          </div>

          <div className="card">
            <div className="text-xs font-semibold mb-2 text-gray-300 uppercase tracking-wide">
              Кумулятивні події в часі (всі напрямки разом)
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={cumulativeData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="t" tick={{ fontSize: 11, fill: '#9ca3af' }} label={{ value: 'sec', offset: -2, position: 'insideBottom', fill: '#9ca3af', fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} />
                <Tooltip contentStyle={{ background: '#161922', border: '1px solid #374151' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Area type="monotone" dataKey="gt" name="Ground Truth" stroke="#facc15" fill="#facc1530" />
                <Area type="monotone" dataKey="predA" name="Метод А" stroke="#3b82f6" fill="transparent" />
                <Area type="monotone" dataKey="predB" name="Метод Б" stroke="#f0b429" fill="transparent" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* Result Presentation for Behavior Mode */}
      {isComplete && evalMode === 'behavior' && jobA?.evaluation && (
        <div className="space-y-6">
          {jobA.evaluation.gt_video_url && videos.length > 0 && (
            <SideBySideVideoPlayer
              videos={[
                { url: jobA.evaluation.gt_video_url, label: 'Ground Truth', borderClass: 'border-yellow-600/60', textClass: 'text-yellow-500' },
                ...videos
              ]}
            />
          )}
          <BehaviorEvalResults result={jobA.evaluation as unknown as import('../types').BehaviorEvalResult} />
        </div>
      )}
    </div>
  );
}
