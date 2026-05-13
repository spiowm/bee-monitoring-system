import { Target } from 'lucide-react';
import type { BehaviorEvalResult } from '../types';

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

export default function BehaviorEvalResults({ result }: Props) {
  // Safe default for gt_events (we don't use it in behavior mode, but keeping structure intact)
  
  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-[#1C1C1E] rounded-xl border border-white/10 p-5 relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-10">
            <Target className="w-16 h-16" />
          </div>
          <div className="relative z-10">
            <h3 className="text-gray-400 font-medium mb-1">Загальна точність (Behavior)</h3>
            <div className="text-3xl font-bold text-white mb-2">
              {(result.overall_accuracy * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">
              {result.total_matched} / {result.total_gt_labeled} GT labels matched
            </div>
          </div>
        </div>

        {result.eval_classes.map((cls) => {
          const metrics = result.per_class[cls];
          if (!metrics) return null;
          
          const color = METRIC_COLORS[cls] || '#A0A0A0';
          const label = METRIC_LABELS[cls] || cls;
          
          return (
            <div key={cls} className="bg-[#1C1C1E] rounded-xl border border-white/10 p-5">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-gray-300 font-medium" style={{ color }}>
                  {label}
                </h3>
                <div className="flex gap-2 text-sm text-gray-500">
                  <span title="GT Count">GT: {metrics.gt_count}</span>
                  <span title="Predicted">Pr: {metrics.pred_count}</span>
                </div>
              </div>
              
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <div className="text-xs text-gray-500 mb-1">F1 Score</div>
                  <div className="text-lg font-bold text-white">{(metrics.f1 * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">Precision</div>
                  <div className="text-lg font-bold text-white">{(metrics.precision * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">Recall</div>
                  <div className="text-lg font-bold text-white">{(metrics.recall * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Confusion Matrix Table */}
      <div className="bg-[#1C1C1E] rounded-xl border border-white/10 overflow-hidden">
        <div className="px-6 py-4 border-b border-white/10">
          <h2 className="text-lg font-semibold text-white">Матриця плутанини (Confusion Matrix)</h2>
          <p className="text-sm text-gray-400 mt-1">Рядки: Фактичний стан (GT) | Стовпці: Передбачений стан</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-white/5 border-b border-white/10">
                <th className="px-6 py-3 text-sm font-medium text-gray-400 w-1/4 border-r border-white/10">
                  Справжній стан ↓
                </th>
                {Object.keys(result.confusion_matrix[result.eval_classes[0]] || {}).map((predCls) => (
                  <th key={predCls} className="px-6 py-3 text-sm font-medium text-gray-400 text-center">
                    {METRIC_LABELS[predCls] || predCls}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {result.eval_classes.map((gtCls) => (
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
    </div>
  );
}
