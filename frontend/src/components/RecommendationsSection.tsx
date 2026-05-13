import { useState } from 'react';
import { Lightbulb, AlertTriangle, AlertCircle, Info, ChevronDown, ChevronRight } from 'lucide-react';
import type { Recommendation } from '../types';

interface Props {
  recommendations?: Recommendation[];
}

const SEVERITY_STYLES: Record<Recommendation['severity'], {
  borderClass: string;
  badgeClass: string;
  Icon: typeof Info;
  label: string;
}> = {
  info: {
    borderClass: 'border-l-gray-500 bg-gray-800/30',
    badgeClass: 'bg-gray-700 text-gray-200',
    Icon: Info,
    label: 'Інформація',
  },
  warning: {
    borderClass: 'border-l-yellow-500 bg-yellow-900/15',
    badgeClass: 'bg-yellow-900/50 text-yellow-300',
    Icon: AlertTriangle,
    label: 'Увага',
  },
  critical: {
    borderClass: 'border-l-red-500 bg-red-900/15',
    badgeClass: 'bg-red-900/50 text-red-300',
    Icon: AlertCircle,
    label: 'Критично',
  },
};

const SEVERITY_ORDER: Record<Recommendation['severity'], number> = {
  critical: 0, warning: 1, info: 2,
};

function DetailsBadge({ details }: { details?: Record<string, number | string> | null }) {
  const [open, setOpen] = useState(false);
  if (!details || Object.keys(details).length === 0) return null;

  return (
    <div className="mt-1.5">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
      >
        {open ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
        Деталі
      </button>
      {open && (
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] font-mono text-gray-500">
          {Object.entries(details).map(([k, v]) => (
            <span key={k}>
              <span className="text-gray-600">{k}:</span>{' '}
              <span className="text-gray-300">{typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function RecommendationsSection({ recommendations }: Props) {
  const recs = (recommendations ?? []).slice().sort(
    (a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity],
  );

  return (
    <div className="bg-[var(--bg-panel)] p-4 rounded-xl border border-gray-800 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200 uppercase tracking-wide flex items-center gap-2">
          <Lightbulb size={16} className="text-[var(--accent)]" />
          Рекомендації пасічнику
        </h3>
        <span className="text-xs text-gray-500">{recs.length}</span>
      </div>

      {recs.length === 0 ? (
        <div className="text-sm text-gray-500 italic py-4 text-center">
          Все в нормі — рекомендацій немає 🎉
        </div>
      ) : (
        <div className="space-y-2">
          {recs.map((r, i) => {
            const style = SEVERITY_STYLES[r.severity];
            const SevIcon = style.Icon;
            return (
              <div
                key={i}
                className={`border-l-4 ${style.borderClass} rounded-r-lg p-3 space-y-1`}
              >
                <div className="flex items-start gap-2">
                  <span className="text-xl leading-none mt-0.5">{r.icon}</span>
                  <div className="flex-grow min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`text-[10px] uppercase font-bold px-1.5 py-0.5 rounded ${style.badgeClass} flex items-center gap-1`}>
                        <SevIcon size={10} />
                        {style.label}
                      </span>
                      {r.rule_id && (
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-gray-800 text-gray-500 border border-gray-700">
                          {r.rule_id}
                        </span>
                      )}
                      <span className="text-sm font-semibold text-gray-100">{r.title}</span>
                    </div>
                    <p className="text-xs text-gray-300 mt-1 leading-relaxed">{r.description}</p>
                    {r.action && (
                      <p className="text-xs italic text-gray-400 mt-1">→ {r.action}</p>
                    )}
                    <DetailsBadge details={r.details} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
