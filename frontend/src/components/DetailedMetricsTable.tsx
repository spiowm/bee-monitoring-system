import { Info } from 'lucide-react';

interface DirMetrics {
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

interface EvalSummary {
  accuracy: number;
  in: DirMetrics;
  out: DirMetrics;
  fps?: number; // Processing FPS if available
}

interface Props {
  a: EvalSummary;
  b?: EvalSummary;
  labelA?: string;
  labelB?: string;
}

interface RowConfig {
  label: string;
  valA: number;
  valB?: number;
  isPct?: boolean;
  higherIsBetter?: boolean;
  isHeader?: boolean;
}

function TableRow({ label, valA, valB, isPct = false, higherIsBetter = true, isHeader = false }: RowConfig) {
  if (isHeader) {
    return (
      <tr className="bg-gray-800/40">
        <td colSpan={valB !== undefined ? 3 : 2} className="px-4 py-2 text-xs font-bold text-gray-400 uppercase tracking-wider">
          {label}
        </td>
      </tr>
    );
  }

  const fmt = (v: number) => (isPct ? `${(v * 100).toFixed(1)}%` : v.toString());
  
  let winA = false;
  let winB = false;
  
  if (valB !== undefined && valA !== valB) {
    if (higherIsBetter) {
      winA = valA > valB;
      winB = valB > valA;
    } else {
      winA = valA < valB;
      winB = valB < valA;
    }
  }

  return (
    <tr className="border-b border-gray-800/50 hover:bg-gray-800/20 transition-colors">
      <td className="px-4 py-2.5 text-sm text-gray-300">{label}</td>
      <td className={`px-4 py-2.5 text-center font-mono text-sm ${winA ? 'text-[var(--accent)] font-bold' : 'text-gray-400'}`}>
        {fmt(valA)}
      </td>
      {valB !== undefined && (
        <td className={`px-4 py-2.5 text-center font-mono text-sm ${winB ? 'text-[var(--accent)] font-bold' : 'text-gray-400'}`}>
          {fmt(valB)}
        </td>
      )}
    </tr>
  );
}

export default function DetailedMetricsTable({ a, b, labelA = "Метод А", labelB = "Метод Б" }: Props) {
  const isCompare = !!b;

  const rows: RowConfig[] = [
    { label: "Загальні метрики", valA: 0, isHeader: true },
    { label: "Accuracy (Загальна точність)", valA: a.accuracy, valB: b?.accuracy, isPct: true, higherIsBetter: true },
    { label: "MAE (Загальна похибка підрахунку)", valA: a.in.mae + a.out.mae, valB: b ? b.in.mae + b.out.mae : undefined, isPct: false, higherIsBetter: false },
    ...(a.fps ? [{ label: "Швидкість обробки (FPS)", valA: a.fps, valB: b?.fps, isPct: false, higherIsBetter: true }] : []),
    
    { label: "Напрямок: ВЛІТ (IN)", valA: 0, isHeader: true },
    { label: "F1 Score", valA: a.in.f1, valB: b?.in.f1, isPct: true, higherIsBetter: true },
    { label: "Precision (Точність детекцій)", valA: a.in.precision, valB: b?.in.precision, isPct: true, higherIsBetter: true },
    { label: "Recall (Повнота знаходжень)", valA: a.in.recall, valB: b?.in.recall, isPct: true, higherIsBetter: true },
    { label: "True Positives (TP) — Знайдено", valA: a.in.tp, valB: b?.in.tp, isPct: false, higherIsBetter: true },
    { label: "False Positives (FP) — Хибні (зайві)", valA: a.in.fp, valB: b?.in.fp, isPct: false, higherIsBetter: false },
    { label: "False Negatives (FN) — Пропущені", valA: a.in.fn, valB: b?.in.fn, isPct: false, higherIsBetter: false },
    
    { label: "Напрямок: ВИЛІТ (OUT)", valA: 0, isHeader: true },
    { label: "F1 Score", valA: a.out.f1, valB: b?.out.f1, isPct: true, higherIsBetter: true },
    { label: "Precision (Точність детекцій)", valA: a.out.precision, valB: b?.out.precision, isPct: true, higherIsBetter: true },
    { label: "Recall (Повнота знаходжень)", valA: a.out.recall, valB: b?.out.recall, isPct: true, higherIsBetter: true },
    { label: "True Positives (TP) — Знайдено", valA: a.out.tp, valB: b?.out.tp, isPct: false, higherIsBetter: true },
    { label: "False Positives (FP) — Хибні (зайві)", valA: a.out.fp, valB: b?.out.fp, isPct: false, higherIsBetter: false },
    { label: "False Negatives (FN) — Пропущені", valA: a.out.fn, valB: b?.out.fn, isPct: false, higherIsBetter: false },
  ];

  return (
    <div className="card space-y-3">
      <div className="flex items-center gap-2 text-[var(--accent)] font-semibold uppercase tracking-wider text-sm mb-2">
        <Info size={18} />
        Детальна таблиця метрик
      </div>
      
      <div className="overflow-x-auto rounded-xl border border-gray-800">
        <table className="w-full text-left border-collapse">
          <thead className="bg-gray-900/80">
            <tr>
              <th className="px-4 py-3 text-xs font-semibold text-gray-400 uppercase">Метрика</th>
              <th className="px-4 py-3 text-center text-xs font-semibold text-blue-400 uppercase">{labelA}</th>
              {isCompare && (
                <th className="px-4 py-3 text-center text-xs font-semibold text-[var(--accent)] uppercase">{labelB}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <TableRow key={i} {...r} />
            ))}
          </tbody>
        </table>
      </div>
      
      {isCompare && (
        <div className="text-xs text-gray-500 mt-2 flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[var(--accent)]" /> 
          <span>Виділено кращий результат (більше для TP/Precision/Recall/F1, менше для FP/FN/MAE).</span>
        </div>
      )}
    </div>
  );
}
