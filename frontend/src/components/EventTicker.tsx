import type { LiveStats } from '../types';

type RecentEvent = NonNullable<LiveStats['recent_events']>[number];

interface Props {
  events: RecentEvent[];
}

const BEHAVIOR_LABELS: Record<string, string> = {
  foraging: 'фуражування',
  fanning: 'вентиляція',
  washboarding: 'полірування',
  defense: 'захист',
};

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function EventTicker({ events }: Props) {
  // Newest first
  const ordered = [...events].reverse();

  return (
    <div>
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 flex items-center justify-between">
        <span>Останні події</span>
        <span className="text-gray-600 normal-case font-normal text-[10px]">{events.length}/10</span>
      </div>
      {ordered.length === 0 ? (
        <div className="bg-[var(--bg-panel)] border border-gray-800 rounded-lg px-3 py-4 text-center text-xs text-gray-600">
          Очікуємо першу подію…
        </div>
      ) : (
        <ul className="space-y-1 max-h-56 overflow-y-auto">
          {ordered.map((e, i) => {
            const isIn = e.direction === 'IN';
            const dirColor = isIn ? 'var(--color-in)' : 'var(--color-out)';
            return (
              <li
                key={`${e.frame}-${e.track_id}-${i}`}
                className="bg-[var(--bg-panel)] border border-gray-800 rounded-md px-2 py-1.5 flex items-center gap-2 text-xs animate-in fade-in slide-in-from-top-2"
                style={{ borderLeftColor: dirColor, borderLeftWidth: 3 }}
              >
                <span className="text-gray-500 font-mono w-10 shrink-0">{formatTime(e.timestamp_sec)}</span>
                <span className="font-bold w-10 shrink-0" style={{ color: dirColor }}>
                  {isIn ? 'IN' : 'OUT'}
                </span>
                <span className="text-gray-400 font-mono w-12 shrink-0">#{e.track_id}</span>
                {e.behavior_class && (
                  <span className="text-[11px] text-gray-400 truncate ml-auto">
                    {BEHAVIOR_LABELS[e.behavior_class] || e.behavior_class}
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
