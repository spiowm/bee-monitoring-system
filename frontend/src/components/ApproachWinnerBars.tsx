import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { Trophy } from 'lucide-react';

interface DirMetrics {
  tp: number; fp: number; fn: number;
  precision: number; recall: number; f1: number; mae: number;
}

interface EvalSummary {
  accuracy: number;
  in: DirMetrics;
  out: DirMetrics;
}

interface Props {
  a: EvalSummary;
  b: EvalSummary;
}

interface MetricRow {
  metric: string;
  A: number;
  B: number;
  /** true якщо більше = краще */
  higherIsBetter: boolean;
  /** для відображення */
  format: (n: number) => string;
}

function buildRows(a: EvalSummary, b: EvalSummary): MetricRow[] {
  const pct = (n: number) => `${(n * 100).toFixed(1)}%`;
  return [
    { metric: 'Accuracy', A: a.accuracy, B: b.accuracy, higherIsBetter: true, format: pct },
    { metric: 'F1 IN',    A: a.in.f1,    B: b.in.f1,    higherIsBetter: true, format: pct },
    { metric: 'F1 OUT',   A: a.out.f1,   B: b.out.f1,   higherIsBetter: true, format: pct },
    {
      metric: 'MAE',
      A: a.in.mae + a.out.mae,
      B: b.in.mae + b.out.mae,
      higherIsBetter: false,
      format: (n) => `±${n}`,
    },
  ];
}

function winnerOf(row: MetricRow): 'A' | 'B' | 'tie' {
  if (row.A === row.B) return 'tie';
  if (row.higherIsBetter) return row.A > row.B ? 'A' : 'B';
  return row.A < row.B ? 'A' : 'B';
}

export default function ApproachWinnerBars({ a, b }: Props) {
  const rows = buildRows(a, b);
  const winners = rows.map(winnerOf);
  const bWins = winners.filter(w => w === 'B').length;
  const aWins = winners.filter(w => w === 'A').length;
  const ties = winners.filter(w => w === 'tie').length;

  // Дані для відображення Recharts (нормалізовано до 0–1, але MAE розгортаємо інакше).
  const chartData = rows.map(r => {
    if (r.metric === 'MAE') {
      const max = Math.max(r.A, r.B, 1);
      return { metric: r.metric, A: 1 - r.A / max, B: 1 - r.B / max, raw_a: r.A, raw_b: r.B };
    }
    return { metric: r.metric, A: r.A, B: r.B, raw_a: r.A, raw_b: r.B };
  });

  const conclusion = (() => {
    if (bWins > aWins) return `Метод Б виграє у ${bWins} з 4 метрик — точніший на тестових даних.`;
    if (aWins > bWins) return `Метод А виграє у ${aWins} з 4 метрик — простіший варіант справляється краще.`;
    return ties === 4 ? 'Обидва методи показали однакові результати.' : 'Методи розділили перевагу порівну.';
  })();

  return (
    <div className="space-y-4">
      {/* Список метрик з переможцем */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {rows.map((r, i) => {
          const w = winners[i];
          return (
            <div
              key={r.metric}
              className="bg-[var(--bg-panel)] border border-gray-800 rounded-lg p-3 flex items-center gap-3"
            >
              <div className="text-xs uppercase font-semibold text-gray-400 w-20 shrink-0">
                {r.metric}
              </div>
              <div className="flex-grow grid grid-cols-2 gap-3 text-sm">
                <div className={`flex items-center gap-1.5 ${w === 'A' ? 'text-[var(--accent)] font-bold' : 'text-gray-400'}`}>
                  {w === 'A' && <Trophy size={12} />}
                  <span>А: {r.format(r.A)}</span>
                </div>
                <div className={`flex items-center gap-1.5 ${w === 'B' ? 'text-[var(--accent)] font-bold' : 'text-gray-400'}`}>
                  {w === 'B' && <Trophy size={12} />}
                  <span>Б: {r.format(r.B)}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Bar chart нормалізовано (MAE інвертовано: вище = менша помилка) */}
      <div className="bg-[var(--bg-panel)] border border-gray-800 rounded-xl p-3">
        <div className="text-xs font-semibold mb-2 text-gray-300 uppercase tracking-wide">
          Метрики порівняння (вище = краще)
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="metric" tick={{ fontSize: 11, fill: '#9ca3af' }} />
            <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11, fill: '#9ca3af' }} />
            <Tooltip
              contentStyle={{ background: '#161922', border: '1px solid #374151' }}
              formatter={(_v, name, ctx) => {
                const item = (ctx?.payload as { raw_a?: number; raw_b?: number }) || {};
                const raw = name === 'A' ? item.raw_a : item.raw_b;
                return [name === 'A' ? `А: ${raw}` : `Б: ${raw}`, ''];
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="A" name="Метод А" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="B" name="Метод Б" fill="#f0b429" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-[var(--accent)]/10 border-l-4 border-[var(--accent)] p-3 rounded text-sm text-gray-200">
        <strong className="text-[var(--accent)]">Висновок: </strong>{conclusion}
      </div>
    </div>
  );
}
