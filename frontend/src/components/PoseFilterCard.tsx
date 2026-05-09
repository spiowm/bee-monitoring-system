import { Filter, CheckCircle2, ShieldOff, XCircle } from 'lucide-react';

interface Props {
  approach: string;
  poseConfirmed: number;
  fallback: number;
  rejected: number;
  rejectReasons?: Record<string, number>;
  angleHistogram?: Record<string, number>;
}

/**
 * Показує ефективність pose-фільтра у Approach B:
 * скільки перетинів зарахував би Approach A vs скільки залишив B + breakdown.
 * Для Approach A — нічого корисного немає, ховаємо.
 */
export default function PoseFilterCard({
  approach,
  poseConfirmed,
  fallback,
  rejected,
  rejectReasons,
  angleHistogram,
}: Props) {
  if (approach !== 'B') return null;

  const counted = poseConfirmed + fallback;
  const wouldCountByA = counted + rejected;
  const rejectedPct = wouldCountByA > 0 ? (rejected / wouldCountByA) * 100 : 0;
  const confirmedPct = wouldCountByA > 0 ? (poseConfirmed / wouldCountByA) * 100 : 0;
  const fallbackPct = wouldCountByA > 0 ? (fallback / wouldCountByA) * 100 : 0;

  // Rejection breakdown
  const angleMismatch = rejectReasons?.angle_mismatch ?? 0;
  const noKeypoints = rejectReasons?.no_keypoints ?? 0;
  // If we have reasons, use them, otherwise fallback to assuming all were angle_mismatch for legacy
  const totalReasons = angleMismatch + noKeypoints || rejected || 1;
  const angleMismatchPct = (angleMismatch / totalReasons) * 100;
  const noKeypointsPct = (noKeypoints / totalReasons) * 100;

  // Histogram
  const maxHistVal = angleHistogram ? Math.max(...Object.values(angleHistogram)) : 0;

  return (
    <div className="bg-[var(--bg-panel)] p-4 rounded-xl border border-blue-900/40 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-blue-300 uppercase tracking-wide flex items-center gap-2">
          <Filter size={16} /> Ефективність pose-фільтра
        </h3>
        <span className="text-xs text-gray-500">Approach B</span>
      </div>

      {/* Великі числа: A would count vs B counted */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
          <div className="text-[11px] text-gray-400 uppercase tracking-wide mb-1">
            Без фільтра (Approach A)
          </div>
          <div className="text-3xl font-bold font-mono text-gray-200">{wouldCountByA}</div>
          <div className="text-xs text-gray-500 mt-1">перетинів зарахував би</div>
        </div>
        <div className="bg-blue-900/20 rounded-lg p-3 border border-blue-700/40">
          <div className="text-[11px] text-blue-300 uppercase tracking-wide mb-1">
            З pose-фільтром (Approach B)
          </div>
          <div className="text-3xl font-bold font-mono text-blue-300">{counted}</div>
          <div className="text-xs text-gray-400 mt-1">
            відсіяно <span className="text-yellow-400 font-semibold">{rejected}</span> хибних
          </div>
        </div>
      </div>

      {/* Composition bar */}
      {wouldCountByA > 0 && (
        <div className="space-y-2">
          <div className="flex h-3 rounded-full overflow-hidden bg-gray-800">
            <div
              className="bg-blue-500"
              style={{ width: `${confirmedPct}%` }}
              title={`Pose-confirmed: ${poseConfirmed} (${confirmedPct.toFixed(1)}%)`}
            />
            <div
              className="bg-gray-500"
              style={{ width: `${fallbackPct}%` }}
              title={`Trajectory fallback: ${fallback} (${fallbackPct.toFixed(1)}%)`}
            />
            <div
              className="bg-yellow-500"
              style={{ width: `${rejectedPct}%` }}
              title={`Rejected: ${rejected} (${rejectedPct.toFixed(1)}%)`}
            />
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
            <span className="flex items-center gap-1.5 text-blue-400">
              <CheckCircle2 size={12} />
              Pose-confirmed: <span className="font-mono font-semibold">{poseConfirmed}</span>
            </span>
            <span className="flex items-center gap-1.5 text-gray-400">
              <ShieldOff size={12} />
              Fallback: <span className="font-mono font-semibold">{fallback}</span>
            </span>
            <span className="flex items-center gap-1.5 text-yellow-400">
              <XCircle size={12} />
              Відсіяно: <span className="font-mono font-semibold">{rejected}</span>
            </span>
          </div>
        </div>
      )}

      {/* Rejection Reasons Analysis (Scientific Detail) */}
      {rejected > 0 && (
        <div className="pt-2 border-t border-gray-800 space-y-2">
          <div className="text-[11px] text-gray-500 uppercase tracking-wider font-semibold">
            Аналіз причин відхилення
          </div>
          <div className="flex h-1.5 rounded-full overflow-hidden bg-gray-800">
            <div
              className="bg-orange-500"
              style={{ width: `${angleMismatchPct}%` }}
              title={`Angle mismatch: ${angleMismatch}`}
            />
            <div
              className="bg-red-900"
              style={{ width: `${noKeypointsPct}%` }}
              title={`No keypoints: ${noKeypoints}`}
            />
          </div>
          <div className="grid grid-cols-2 gap-2 text-[10px]">
            <div className="flex items-center gap-2 text-orange-400/80">
              <div className="w-2 h-2 rounded-full bg-orange-500" />
              <span>Невідповідність кута (±60°):</span>
              <span className="text-white font-mono ml-auto">{angleMismatch}</span>
            </div>
            <div className="flex items-center gap-2 text-red-400/80">
              <div className="w-2 h-2 rounded-full bg-red-900" />
              <span>Відсутні keypoints (голова/жало):</span>
              <span className="text-white font-mono ml-auto">{noKeypoints}</span>
            </div>
          </div>
        </div>
      )}

      {/* Angle Histogram */}
      {angleHistogram && maxHistVal > 0 && (
        <div className="pt-3 border-t border-gray-800">
          <div className="text-[11px] text-gray-500 uppercase tracking-wider font-semibold mb-3">
            Розподіл кутових розбіжностей
          </div>
          <div className="flex items-end gap-1.5 h-16 mt-2">
            {Object.entries(angleHistogram).map(([bin, count]) => {
              const isRejectedBin = bin === '>60';
              const heightPct = count > 0 ? (count / maxHistVal) * 100 : 0;
              return (
                <div key={bin} className="flex-1 flex flex-col items-center gap-1 group">
                  <div className="text-[9px] font-mono text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity">
                    {count}
                  </div>
                  <div 
                    className={`w-full rounded-t-sm ${isRejectedBin ? 'bg-orange-500' : 'bg-blue-500'}`} 
                    style={{ height: `${heightPct}%`, minHeight: count > 0 ? '4px' : '0' }} 
                    title={`${bin}°: ${count} подій`}
                  />
                  <div className="text-[9px] text-gray-500">{bin}°</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <p className="text-[11px] text-gray-500 italic leading-relaxed pt-1">
        Pose-фільтр відсіює перетини, де вектор голова→жало не співпадає з напрямком руху.
        Більшість відхилень через "невідповідність кута" свідчать про хаотичний рух (джиттер) на лінії, 
        що не є справжнім вльотом/вильотом.
      </p>
    </div>
  );
}
