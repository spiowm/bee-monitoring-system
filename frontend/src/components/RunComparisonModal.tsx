import { X } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis,
} from 'recharts';
import type { Job } from '../types';

const JOB_COLORS = ['#f0b429', '#63b3ed', '#48bb78', '#fc8181'];

const TOOLTIP_STYLE = {
  background: 'var(--bg-card)',
  border: '1px solid #4a5568',
  borderRadius: '8px',
  fontSize: '12px',
};

const BEHAVIOR_KEYS = ['foraging', 'fanning', 'guarding', 'washboarding'];

export default function RunComparisonModal({ jobs, onClose }: { jobs: Job[]; onClose: () => void }) {
  const label = (j: Job, i: number) => `Run ${i + 1}: ${j.filename.split('.')[0].slice(0, 16)}`;

  // Grouped bar: IN / OUT per job
  const trafficData = [
    { metric: 'IN', ...Object.fromEntries(jobs.map((j, i) => [label(j, i), j.result?.total_in ?? 0])) },
    { metric: 'OUT', ...Object.fromEntries(jobs.map((j, i) => [label(j, i), j.result?.total_out ?? 0])) },
  ];

  // Behavior grouped bar
  const behaviorData = BEHAVIOR_KEYS.map(bk => ({
    behavior: bk,
    ...Object.fromEntries(jobs.map((j, i) => [
      label(j, i),
      (j.result?.behavior_summary as any)?.[`${bk}_detections`] ?? 0,
    ])),
  }));

  // Radar: normalized metrics per run
  const maxIn = Math.max(1, ...jobs.map(j => j.result?.total_in ?? 0));
  const maxOut = Math.max(1, ...jobs.map(j => j.result?.total_out ?? 0));
  const maxFps = Math.max(1, ...jobs.map(j => j.result?.fps_processed ?? 0));
  const maxPose = Math.max(1, ...jobs.map(j => j.result?.pose_confirmed_events ?? 0));

  const radarData = [
    { metric: 'IN', ...Object.fromEntries(jobs.map((j, i) => [label(j, i), Math.round((j.result?.total_in ?? 0) / maxIn * 100)])) },
    { metric: 'OUT', ...Object.fromEntries(jobs.map((j, i) => [label(j, i), Math.round((j.result?.total_out ?? 0) / maxOut * 100)])) },
    { metric: 'FPS', ...Object.fromEntries(jobs.map((j, i) => [label(j, i), Math.round((j.result?.fps_processed ?? 0) / maxFps * 100)])) },
    { metric: 'Pose ✓', ...Object.fromEntries(jobs.map((j, i) => [label(j, i), Math.round((j.result?.pose_confirmed_events ?? 0) / maxPose * 100)])) },
    { metric: 'Balance', ...Object.fromEntries(jobs.map((j, i) => {
        const bal = (j.result?.total_in ?? 0) - (j.result?.total_out ?? 0);
        return [label(j, i), Math.max(0, Math.round(50 + bal))];
      })) },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <div className="bg-[var(--bg-card)] rounded-xl w-full max-w-5xl max-h-[92vh] overflow-y-auto border border-gray-700 shadow-2xl">

        {/* Header */}
        <div className="sticky top-0 bg-[var(--bg-card)] border-b border-gray-800 p-4 flex justify-between items-center z-10">
          <h2 className="text-base font-bold">Run Comparison ({jobs.length} сесії)</h2>
          <button onClick={onClose} className="p-1.5 hover:bg-gray-800 rounded-full transition-colors">
            <X size={18} />
          </button>
        </div>

        <div className="p-4 space-y-5">

          {/* Job cards side by side */}
          <div className={`grid gap-3 ${jobs.length <= 2 ? 'grid-cols-2' : jobs.length === 3 ? 'grid-cols-3' : 'grid-cols-4'}`}>
            {jobs.map((j, i) => {
              const poseRate = j.result
                ? ((j.result.pose_confirmed_events / Math.max(1, j.result.pose_confirmed_events + j.result.fallback_events)) * 100)
                : 0;
              return (
                <div key={j.job_id} className="bg-[var(--bg-panel)] rounded-xl border p-3 space-y-2" style={{ borderColor: JOB_COLORS[i] + '80' }}>
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full shrink-0" style={{ background: JOB_COLORS[i] }} />
                    <span className="text-xs font-bold text-gray-200">Run {i + 1}</span>
                  </div>
                  <p className="text-xs text-gray-400 truncate" title={j.filename}>{j.filename}</p>
                  <div className="text-xs text-gray-500 space-y-0.5">
                    <div>Approach: <span className="text-gray-200">{j.result?.approach_used || j.config?.approach}</span></div>
                    <div>Tracker: <span className="text-gray-200">{j.result?.tracker_used || j.config?.tracker_name}</span></div>
                    {(j.config as any)?.model_name && (
                      <div>Model: <span className="text-[var(--accent)]">{(j.config as any).model_name}</span></div>
                    )}
                  </div>
                  <div className="grid grid-cols-2 gap-1 pt-1 border-t border-gray-800">
                    <div className="text-center">
                      <div className="text-lg font-bold text-[var(--color-in)]">{j.result?.total_in ?? '—'}</div>
                      <div className="text-xs text-gray-500">IN</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-[var(--color-out)]">{j.result?.total_out ?? '—'}</div>
                      <div className="text-xs text-gray-500">OUT</div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm font-bold text-gray-200">{j.result?.fps_processed?.toFixed(1) ?? '—'}</div>
                      <div className="text-xs text-gray-500">FPS</div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm font-bold text-blue-400">{poseRate.toFixed(0)}%</div>
                      <div className="text-xs text-gray-500">Pose ✓</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Charts row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Traffic comparison */}
            <div className="bg-[var(--bg-panel)] p-3 rounded-xl border border-gray-800">
              <div className="text-xs font-semibold text-gray-300 uppercase tracking-wide mb-2">Traffic IN / OUT</div>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={trafficData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                  <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  {jobs.map((j, i) => (
                    <Bar key={j.job_id} dataKey={label(j, i)} fill={JOB_COLORS[i]} radius={[2, 2, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Behavior comparison */}
            <div className="bg-[var(--bg-panel)] p-3 rounded-xl border border-gray-800">
              <div className="text-xs font-semibold text-gray-300 uppercase tracking-wide mb-2">Behavior Distribution</div>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={behaviorData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                  <XAxis dataKey="behavior" tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  {jobs.map((j, i) => (
                    <Bar key={j.job_id} dataKey={label(j, i)} fill={JOB_COLORS[i]} radius={[2, 2, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Radar: normalized metrics */}
            <div className="bg-[var(--bg-panel)] p-3 rounded-xl border border-gray-800">
              <div className="text-xs font-semibold text-gray-300 uppercase tracking-wide mb-2">Normalized Overview</div>
              <ResponsiveContainer width="100%" height={160}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#2d3748" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 9, fill: '#a0aec0' }} />
                  {jobs.map((j, i) => (
                    <Radar key={j.job_id} name={`Run ${i + 1}`} dataKey={label(j, i)} stroke={JOB_COLORS[i]} fill={JOB_COLORS[i]} fillOpacity={0.15} />
                  ))}
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
