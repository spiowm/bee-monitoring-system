import { useState } from 'react';
import { ChevronDown, ChevronRight, Info } from 'lucide-react';
import type { LiveStats, Job } from '../types';
import EventTicker from './EventTicker';
import BehaviorLegend from './BehaviorLegend';
import PoseFilterCard from './PoseFilterCard';

const BEHAVIORS = [
  { key: 'foraging',     label: 'Фуражування',  color: 'var(--behavior-foraging)' },
  { key: 'fanning',      label: 'Вентиляція',    color: 'var(--behavior-fanning)' },
  { key: 'washboarding', label: 'Полірування',   color: 'var(--behavior-washboarding)' },
  { key: 'defense',      label: 'Захист',        color: 'var(--behavior-defense)' },
] as const;

interface LiveStatsPanelProps {
  liveStats: LiveStats | null;
  job: Job | null;
}

export default function LiveStatsPanel({ liveStats, job }: LiveStatsPanelProps) {
  const [isTimingsOpen, setIsTimingsOpen] = useState(false);

  const totalIn    = liveStats?.total_in    ?? job?.result?.total_in    ?? 0;
  const totalOut   = liveStats?.total_out   ?? job?.result?.total_out   ?? 0;
  const activeBees = liveStats?.bees_on_ramp ?? 0;
  const fps        = liveStats?.current_fps ?? job?.result?.fps_processed ?? 0;
  const poseOk     = liveStats?.pose_confirmed ?? job?.result?.pose_confirmed_events ?? 0;
  const fallback   = liveStats?.fallback_events ?? job?.result?.fallback_events ?? 0;
  const rejected   = liveStats?.pose_rejected ?? job?.result?.pose_rejected_events ?? 0;
  const approach   = liveStats?.approach ?? job?.result?.approach_used ?? 'A';
  const net        = totalIn - totalOut;
  const timings    = liveStats?.timings;

  const behaviorCounts = BEHAVIORS.map(b => ({
    ...b,
    count: liveStats?.behavior_counts?.[b.key]
      ?? (job?.result?.behavior_summary as Record<string, number> | undefined)?.[`${b.key}_detections`]
      ?? 0,
  }));
  const totalBehavior = Math.max(1, behaviorCounts.reduce((s, b) => s + b.count, 0));

  const hasData = liveStats !== null || job !== null;

  return (
    <div className="card h-full flex flex-col gap-4">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider border-b border-gray-800 pb-2">
        Статистика в реальному часі
      </h2>

      {!hasData ? (
        <div className="flex-grow flex items-center justify-center text-gray-600 text-sm">
          Очікування даних…
        </div>
      ) : (
        <>
          {/* В / З / Нетто */}
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-[var(--bg-panel)] p-3 rounded-xl border border-[var(--color-in)]/25 text-center">
              <div className="text-xs text-gray-500 mb-1">В</div>
              <div className="text-2xl font-bold" style={{ color: 'var(--color-in)' }}>{totalIn}</div>
            </div>
            <div className="bg-[var(--bg-panel)] p-3 rounded-xl border border-[var(--color-out)]/25 text-center">
              <div className="text-xs text-gray-500 mb-1">З</div>
              <div className="text-2xl font-bold" style={{ color: 'var(--color-out)' }}>{totalOut}</div>
            </div>
            <div
              className="bg-[var(--bg-panel)] p-3 rounded-xl border border-gray-700 text-center cursor-help"
              title="В мінус З (позитивне = більше влетіло)"
            >
              <div className="text-xs text-gray-500 mb-1">Нетто</div>
              <div
                className="text-2xl font-bold"
                style={{ color: net > 0 ? 'var(--color-in)' : net < 0 ? 'var(--color-out)' : 'var(--text-muted)' }}
              >
                {net > 0 ? `+${net}` : net}
              </div>
            </div>
          </div>

          {/* На рампі */}
          <div className="flex items-center justify-between bg-[var(--bg-panel)] px-4 py-3 rounded-xl border border-gray-700">
            <div className="flex items-center gap-2">
              {activeBees > 0 && (
                <span className="w-2 h-2 rounded-full bg-[var(--accent)] animate-pulse" />
              )}
              <span className="text-sm text-gray-400">На рампі зараз</span>
            </div>
            <span className="text-xl font-bold text-[var(--accent)]">{activeBees}</span>
          </div>

          {/* Метрики пайплайну */}
          <div className={`grid ${approach === 'B' ? 'grid-cols-4' : 'grid-cols-3'} gap-2 text-center text-xs`}>
            <div className="bg-[var(--bg-panel)] rounded-lg py-2 px-1 border border-gray-800">
              <div className="text-gray-500 mb-0.5">Кадрів/с</div>
              <div className="font-mono font-semibold text-gray-200">{fps > 0 ? fps.toFixed(1) : '—'}</div>
            </div>
            <div className="bg-[var(--bg-panel)] rounded-lg py-2 px-1 border border-gray-800">
              <div className="text-gray-500 mb-0.5">Поза ✓</div>
              <div className="font-mono font-semibold" style={{ color: 'var(--color-pose)' }}>{poseOk}</div>
            </div>
            <div className="bg-[var(--bg-panel)] rounded-lg py-2 px-1 border border-gray-800">
              <div className="text-gray-500 mb-0.5">Резерв</div>
              <div className="font-mono font-semibold" style={{ color: 'var(--color-fallback)' }}>{fallback}</div>
            </div>
            {approach === 'B' && (
              <div
                className="bg-[var(--bg-panel)] rounded-lg py-2 px-1 border border-yellow-700/40"
                title="Перетинів, які Approach A зарахував би, але Approach B відсіяв через невідповідність вектора пози"
              >
                <div className="text-gray-500 mb-0.5">Відсіяно</div>
                <div className="font-mono font-semibold text-yellow-400">{rejected}</div>
              </div>
            )}
          </div>

          {/* Pose Filter Card (B only) */}
          <PoseFilterCard
            approach={approach}
            poseConfirmed={poseOk}
            fallback={fallback}
            rejected={rejected}
            rejectReasons={liveStats?.reject_reasons ?? job?.result?.reject_reasons}
            angleHistogram={liveStats?.angle_histogram ?? job?.result?.angle_histogram}
          />

          {/* Поведінкові бари */}
          <div>
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Поведінка</div>
            <div className="space-y-2.5">
              {behaviorCounts.map(b => (
                <div key={b.key}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">{b.label}</span>
                    <span className="font-mono font-medium text-gray-300">{b.count}</span>
                  </div>
                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${(b.count / totalBehavior) * 100}%`,
                        background: b.color,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <BehaviorLegend compact />

          {/* Pipeline Timing Section */}
          {timings && (
            <div className="border border-gray-800 rounded-xl overflow-hidden bg-[var(--bg-panel)] shrink-0">
              <button
                onClick={() => setIsTimingsOpen(!isTimingsOpen)}
                className="w-full flex items-center justify-between p-3 hover:bg-gray-800/50 transition-colors text-xs font-semibold text-gray-400 uppercase tracking-wider"
              >
                <div className="flex items-center gap-2">
                  {isTimingsOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  Таймінги пайплайну
                </div>
                <div className="flex items-center gap-3 font-mono text-gray-500 normal-case">
                  <span title="FPS моделі">🎯 {timings.model_fps?.toFixed(1)}</span>
                  <span title="FPS пайплайну">📹 {timings.pipeline_fps?.toFixed(1)}</span>
                </div>
              </button>

              {isTimingsOpen && (
                <div className="p-4 pt-0 border-t border-gray-800/50 space-y-4 text-xs">
                  <div className="grid grid-cols-2 gap-2 bg-gray-900/50 rounded-lg p-3 border border-gray-800">
                    <div>
                      <div className="flex items-center gap-1.5 text-gray-400 mb-1">
                        🎯 FPS моделі
                        <Info size={12} className="text-gray-500 cursor-help" title="Швидкість обробки нейромережею" />
                      </div>
                      <div className="text-lg font-mono font-bold text-white">{timings.model_fps?.toFixed(1)}</div>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center justify-end gap-1.5 text-gray-400 mb-1">
                        <Info size={12} className="text-gray-500 cursor-help" title="Реальна швидкість з урахуванням малювання та логіки" />
                        FPS пайплайну 📹
                      </div>
                      <div className="text-lg font-mono font-bold text-[var(--accent)]">{timings.pipeline_fps?.toFixed(1)}</div>
                    </div>
                  </div>

                  <div className="flex h-4 w-full rounded-sm overflow-hidden border border-gray-800">
                    <div title="Це час інференсу нейромережі YOLO. Зменшення розміру моделі або роздільної здатності напряму зменшує це значення." className="bg-blue-500 hover:brightness-110 transition-colors cursor-help" style={{ width: `${(timings.detection_ms / timings.total_ms) * 100}%` }} />
                    <div title={`Відстеження: ${timings.tracking_ms?.toFixed(1)}ms`} className="bg-green-500 hover:brightness-110 transition-colors" style={{ width: `${(timings.tracking_ms / timings.total_ms) * 100}%` }} />
                    <div title={`Аналіз поведінки: ${timings.behavior_ms?.toFixed(1)}ms`} className="bg-yellow-500 hover:brightness-110 transition-colors" style={{ width: `${(timings.behavior_ms / timings.total_ms) * 100}%` }} />
                    <div title={`Захисні кластери: ${timings.defense_ms?.toFixed(1)}ms`} className="bg-red-500 hover:brightness-110 transition-colors" style={{ width: `${(timings.defense_ms / timings.total_ms) * 100}%` }} />
                    <div title={`Підрахунок: ${timings.counting_ms?.toFixed(1)}ms`} className="bg-purple-500 hover:brightness-110 transition-colors" style={{ width: `${(timings.counting_ms / timings.total_ms) * 100}%` }} />
                    <div title={`Візуалізація: ${timings.annotation_ms?.toFixed(1)}ms`} className="bg-gray-300 hover:brightness-110 transition-colors" style={{ width: `${(timings.annotation_ms / timings.total_ms) * 100}%` }} />
                  </div>

                  <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-gray-400 font-mono">
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-sm bg-blue-500" />
                      <span>Інференс моделі:</span>
                      <span className="text-white ml-auto">{timings.detection_ms?.toFixed(1)}ms</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-sm bg-green-500" />
                      <span>ByteTrack:</span>
                      <span className="text-white ml-auto">{timings.tracking_ms?.toFixed(1)}ms</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-sm bg-yellow-500" />
                      <span>Аналіз поведінки:</span>
                      <span className="text-white ml-auto">{timings.behavior_ms?.toFixed(1)}ms</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-sm bg-red-500" />
                      <span>Захисні кластери:</span>
                      <span className="text-white ml-auto">{timings.defense_ms?.toFixed(1)}ms</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-sm bg-purple-500" />
                      <span>Підрахунок:</span>
                      <span className="text-white ml-auto">{timings.counting_ms?.toFixed(1)}ms</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-sm bg-gray-300" />
                      <span>Візуалізація:</span>
                      <span className="text-white ml-auto">{timings.annotation_ms?.toFixed(1)}ms</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 font-mono pt-2 border-t border-gray-800">
                    <div className="w-2.5 h-[2px] bg-gray-500" />
                    <span>Всього:</span>
                    <span className="text-[var(--accent)] font-bold ml-auto">{timings.total_ms?.toFixed(1)}ms</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Стрічка подій */}
          <div className="flex-grow flex flex-col min-h-0">
            <EventTicker events={liveStats?.recent_events || []} />
          </div>
        </>
      )}
    </div>
  );
}
