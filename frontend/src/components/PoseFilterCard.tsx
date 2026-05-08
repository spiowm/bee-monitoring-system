import { Filter, CheckCircle2, ShieldOff, XCircle } from 'lucide-react';

interface Props {
  approach: string;
  poseConfirmed: number;
  fallback: number;
  rejected: number;
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
}: Props) {
  if (approach !== 'B') return null;

  const counted = poseConfirmed + fallback;
  const wouldCountByA = counted + rejected;
  const rejectedPct = wouldCountByA > 0 ? (rejected / wouldCountByA) * 100 : 0;
  const confirmedPct = wouldCountByA > 0 ? (poseConfirmed / wouldCountByA) * 100 : 0;
  const fallbackPct = wouldCountByA > 0 ? (fallback / wouldCountByA) * 100 : 0;

  return (
    <div className="bg-[var(--bg-panel)] p-4 rounded-xl border border-blue-900/40 space-y-3">
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
        <div className="space-y-1.5">
          <div className="flex h-2.5 rounded-full overflow-hidden bg-gray-800">
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
              {rejectedPct > 0 && (
                <span className="text-gray-500">({rejectedPct.toFixed(0)}%)</span>
              )}
            </span>
          </div>
        </div>
      )}

      <p className="text-[11px] text-gray-500 italic leading-relaxed">
        Pose-фільтр відсіює перетини, де вектор голова→жало не співпадає з напрямком руху
        (за межами ±60°). Це позбавляє фейкових подій, коли бджолу штовхає чужа траєкторія.
      </p>
    </div>
  );
}
