import type { LiveStats, Job } from '../types';

const BEHAVIORS = [
  { key: 'foraging',     label: 'Фуражування',  color: 'var(--behavior-foraging)' },
  { key: 'fanning',      label: 'Вентиляція',    color: 'var(--behavior-fanning)' },
  { key: 'guarding',     label: 'Охорона',       color: 'var(--behavior-guarding)' },
  { key: 'washboarding', label: 'Вошборд',       color: 'var(--behavior-washboarding)' },
] as const;

interface LiveStatsPanelProps {
  liveStats: LiveStats | null;
  job: Job | null;
}

export default function LiveStatsPanel({ liveStats, job }: LiveStatsPanelProps) {
  const totalIn    = liveStats?.total_in    ?? job?.result?.total_in    ?? 0;
  const totalOut   = liveStats?.total_out   ?? job?.result?.total_out   ?? 0;
  const activeBees = liveStats?.bees_on_ramp ?? 0;
  const fps        = liveStats?.current_fps ?? job?.result?.fps_processed ?? 0;
  const poseOk     = liveStats?.pose_confirmed ?? job?.result?.pose_confirmed_events ?? 0;
  const fallback   = liveStats?.fallback_events ?? job?.result?.fallback_events ?? 0;
  const net        = totalIn - totalOut;

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
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
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
          </div>

          {/* Поведінкові бари */}
          <div className="flex-grow">
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
        </>
      )}
    </div>
  );
}
