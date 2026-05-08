interface DirectionMetrics {
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
  mae: number;
  gt_count: number;
  pred_count: number;
}

interface Props {
  accuracy: number;
  in_metrics: DirectionMetrics;
  out_metrics: DirectionMetrics;
}

interface CardProps {
  title: string;
  value: string;
  subtitle?: string;
  borderClass: string;
  textClass?: string;
  hero?: boolean;
}

function KpiCard({ title, value, subtitle, borderClass, textClass, hero }: CardProps) {
  return (
    <div
      className={`bg-[var(--bg-panel)] rounded-xl border ${borderClass} ${
        hero ? 'p-6 md:p-7' : 'p-5'
      } transition-shadow hover:shadow-lg`}
    >
      <div
        className={`text-xs font-semibold mb-3 uppercase tracking-widest ${
          textClass ?? 'text-gray-300'
        }`}
      >
        {title}
      </div>
      <div className={`font-bold font-mono ${hero ? 'text-5xl md:text-6xl' : 'text-4xl'}`}>
        {value}
      </div>
      {subtitle && <div className="text-sm text-gray-400 mt-2">{subtitle}</div>}
    </div>
  );
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export default function EvaluationKPICards({ accuracy, in_metrics, out_metrics }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <KpiCard
        hero
        title="Загальна точність системи"
        value={pct(accuracy)}
        subtitle={`${in_metrics.tp + out_metrics.tp} TP проти ground truth`}
        borderClass="border-2 border-[var(--accent)] shadow-[0_0_25px_rgba(240,180,41,0.12)]"
        textClass="text-[var(--accent)]"
      />
      <KpiCard
        hero
        title="MAE — сумарна помилка підрахунку"
        value={`±${in_metrics.mae + out_metrics.mae}`}
        subtitle={`IN: ±${in_metrics.mae} бджіл · OUT: ±${out_metrics.mae} бджіл`}
        borderClass="border-2 border-blue-500/40 shadow-[0_0_15px_rgba(59,130,246,0.08)]"
        textClass="text-blue-400"
      />
      <KpiCard
        title="F1-score: вліт (IN)"
        value={pct(in_metrics.f1)}
        subtitle={`Precision ${pct(in_metrics.precision)} · Recall ${pct(in_metrics.recall)}`}
        borderClass="border border-green-900/50"
        textClass="text-green-400"
      />
      <KpiCard
        title="F1-score: виліт (OUT)"
        value={pct(out_metrics.f1)}
        subtitle={`Precision ${pct(out_metrics.precision)} · Recall ${pct(out_metrics.recall)}`}
        borderClass="border border-red-900/50"
        textClass="text-red-400"
      />
    </div>
  );
}
