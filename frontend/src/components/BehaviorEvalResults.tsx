import { Target, Zap } from 'lucide-react';
import type { BehaviorEvalResult, BehaviorClassMetrics } from '../types';

interface Props {
  result: BehaviorEvalResult;
}

const METRIC_COLORS: Record<string, string> = {
  foraging: '#FFB020',
  fanning: '#33C2FF',
  washboarding: '#E354F4',
  defense: '#FF5C5C',
};

const METRIC_LABELS: Record<string, string> = {
  foraging: 'Фуражування',
  fanning: 'Вентиляція',
  washboarding: 'Полірування',
  defense: 'Захист',
};

function ClassCard({ cls, metrics }: { cls: string; metrics: BehaviorClassMetrics }) {
  const color = METRIC_COLORS[cls] || '#A0A0A0';
  const label = METRIC_LABELS[cls] || cls;
  return (
    <div className="bg-[#1C1C1E] rounded-xl border border-white/10 p-5">
      <div className="flex justify-between items-start mb-4">
        <h3 className="font-medium" style={{ color }}>{label}</h3>
        <div className="flex gap-2 text-sm text-gray-500">
          <span title="GT Count">GT: {metrics.gt_count}</span>
          <span title="Predicted">Pr: {metrics.pred_count}</span>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2">
        {(['f1', 'precision', 'recall'] as const).map(key => (
          <div key={key}>
            <div className="text-xs text-gray-500 mb-1">
              {key === 'f1' ? 'F1 Score' : key === 'precision' ? 'Precision' : 'Recall'}
            </div>
            <div className="text-lg font-bold text-white">
              {(metrics[key] * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 flex gap-3 text-xs text-gray-600">
        <span>TP {metrics.tp}</span>
        <span>FP {metrics.fp}</span>
        <span>FN {metrics.fn}</span>
      </div>
    </div>
  );
}

function ForagingCard({ metrics }: { metrics: BehaviorClassMetrics }) {
  return (
    <div className="bg-[#1C1C1E] rounded-xl border border-[#FFB020]/30 p-5">
      <div className="flex justify-between items-start mb-1">
        <h3 className="font-medium flex items-center gap-2" style={{ color: '#FFB020' }}>
          <Zap size={14} /> Фуражування
        </h3>
        <div className="flex gap-2 text-sm text-gray-500">
          <span>GT-подій: {metrics.gt_count}</span>
          <span>Pred: {metrics.pred_count}</span>
        </div>
      </div>
      <p className="text-xs text-gray-500 mb-4">Подієва метрика — counting events (±15 кадрів)</p>
      <div className="grid grid-cols-3 gap-2">
        {(['f1', 'precision', 'recall'] as const).map(key => (
          <div key={key}>
            <div className="text-xs text-gray-500 mb-1">
              {key === 'f1' ? 'F1 Score' : key === 'precision' ? 'Precision' : 'Recall'}
            </div>
            <div className="text-lg font-bold text-white">
              {(metrics[key] * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 flex gap-3 text-xs text-gray-600">
        <span>TP {metrics.tp}</span>
        <span>FP {metrics.fp}</span>
        <span>FN {metrics.fn}</span>
      </div>
    </div>
  );
}

export default function BehaviorEvalResults({ result }: Props) {
  const excludedInfo = [
    result.excluded_multilabel != null && result.excluded_multilabel > 0
      ? `${result.excluded_multilabel} multi-label рядків виключено`
      : null,
    result.warmup_frames != null && result.warmup_frames > 0
      ? `warm-up ${result.warmup_frames} кадрів (fanning)`
      : null,
  ].filter(Boolean);

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Overall accuracy */}
        <div className="bg-[#1C1C1E] rounded-xl border border-white/10 p-5 relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-10">
            <Target className="w-16 h-16" />
          </div>
          <div className="relative z-10">
            <h3 className="text-gray-400 font-medium mb-1">Загальна точність</h3>
            <div className="text-3xl font-bold text-white mb-2">
              {(result.overall_accuracy * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">
              {result.total_matched} / {result.total_gt_labeled} GT matched
            </div>
            {excludedInfo.length > 0 && (
              <div className="mt-2 text-xs text-gray-600">
                {excludedInfo.join(' · ')}
              </div>
            )}
          </div>
        </div>

        {/* Per-class cards */}
        {result.eval_classes.map(cls => {
          const metrics = result.per_class[cls];
          if (!metrics) return null;
          return <ClassCard key={cls} cls={cls} metrics={metrics} />;
        })}
      </div>

      {/* Foraging event-based card */}
      {result.foraging_events && (
        <ForagingCard metrics={result.foraging_events} />
      )}

      {/* Confusion Matrix */}
      {result.eval_classes.length > 0 && result.confusion_matrix && (
        <div className="bg-[#1C1C1E] rounded-xl border border-white/10 overflow-hidden">
          <div className="px-6 py-4 border-b border-white/10">
            <h2 className="text-lg font-semibold text-white">Матриця плутанини</h2>
            <p className="text-sm text-gray-400 mt-1">Рядки: Фактичний стан (GT) | Стовпці: Передбачений стан</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-white/5 border-b border-white/10">
                  <th className="px-6 py-3 text-sm font-medium text-gray-400 w-1/4 border-r border-white/10">
                    GT ↓ / Pred →
                  </th>
                  {Object.keys(result.confusion_matrix[result.eval_classes[0]] || {}).map(predCls => (
                    <th key={predCls} className="px-6 py-3 text-sm font-medium text-gray-400 text-center">
                      {METRIC_LABELS[predCls] || predCls}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {result.eval_classes.map(gtCls => (
                  <tr key={gtCls} className="hover:bg-white/[0.02] transition-colors">
                    <td className="px-6 py-4 font-medium border-r border-white/10" style={{ color: METRIC_COLORS[gtCls] || '#A0A0A0' }}>
                      {METRIC_LABELS[gtCls] || gtCls}
                    </td>
                    {Object.entries(result.confusion_matrix[gtCls] || {}).map(([predCls, count]) => {
                      const isDiagonal = gtCls === predCls;
                      const isError = !isDiagonal && count > 0;
                      return (
                        <td
                          key={predCls}
                          className={`px-6 py-4 text-center font-mono ${
                            isDiagonal ? 'text-green-400 font-bold bg-green-400/10' :
                            isError ? 'text-red-400 bg-red-400/5' : 'text-gray-600'
                          }`}
                        >
                          {count}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
