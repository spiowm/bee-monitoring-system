import { useState } from 'react';
import { X, Download } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { getJobJobsJobIdGet } from '../api/generated';
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  AreaChart, Area, Legend,
} from 'recharts';
import type { Job, EventRecord } from '../types';

const BEHAVIOR_COLORS: Record<string, string> = {
  foraging: '#48bb78',
  fanning: '#63b3ed',
  guarding: '#f0b429',
  washboarding: '#a0aec0',
};

const TOOLTIP_STYLE = {
  background: 'var(--bg-card)',
  border: '1px solid #4a5568',
  borderRadius: '8px',
  fontSize: '12px',
};

function SpeedHistogram({ events }: { events: EventRecord[] }) {
  const buckets: Record<number, number> = {};
  events.forEach(e => {
    const bucket = Math.floor(e.speed_px_per_sec / 50) * 50;
    buckets[bucket] = (buckets[bucket] || 0) + 1;
  });
  const data = Object.entries(buckets)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([k, v]) => ({ range: `${k}–${Number(k) + 50}`, count: v }));

  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
        <XAxis dataKey="range" tick={{ fontSize: 9 }} />
        <YAxis tick={{ fontSize: 9 }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} />
        <Bar dataKey="count" fill="var(--accent)" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function AngleHistogram({ events }: { events: EventRecord[] }) {
  const angleEvents = events.filter(e => e.angle_deg !== null);
  if (angleEvents.length === 0) {
    return <div className="h-[150px] flex items-center justify-center text-xs text-gray-500">Немає angle даних (Approach A)</div>;
  }
  const buckets: Record<number, number> = {};
  angleEvents.forEach(e => {
    const bucket = Math.floor(e.angle_deg! / 15) * 15;
    buckets[bucket] = (buckets[bucket] || 0) + 1;
  });
  const data = Object.entries(buckets)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([k, v]) => ({ range: `${k}–${Number(k) + 15}°`, count: v }));

  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
        <XAxis dataKey="range" tick={{ fontSize: 9 }} />
        <YAxis tick={{ fontSize: 9 }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} />
        <Bar dataKey="count" fill="var(--color-pose)" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function CumulativeTimeline({ events }: { events: EventRecord[] }) {
  let cumIn = 0, cumOut = 0;
  const data = [...events]
    .sort((a, b) => a.timestamp_sec - b.timestamp_sec)
    .map(e => {
      if (e.direction === 'IN') cumIn++; else cumOut++;
      return { time: e.timestamp_sec.toFixed(0), IN: cumIn, OUT: cumOut };
    });

  return (
    <ResponsiveContainer width="100%" height={150}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
        <XAxis dataKey="time" tick={{ fontSize: 9 }} unit="s" />
        <YAxis tick={{ fontSize: 9 }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Area type="monotone" dataKey="IN" stroke="var(--color-in)" fill="var(--color-in)" fillOpacity={0.2} strokeWidth={2} />
        <Area type="monotone" dataKey="OUT" stroke="var(--color-out)" fill="var(--color-out)" fillOpacity={0.2} strokeWidth={2} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export default function JobDetailModal({ jobId, onClose }: { jobId: string; onClose: () => void }) {
  const [dirFilter, setDirFilter] = useState<'ALL' | 'IN' | 'OUT'>('ALL');
  const [methodFilter, setMethodFilter] = useState<string>('ALL');

  const { data: job, isLoading } = useQuery({
    queryKey: ['jobDetail', jobId],
    queryFn: async () => {
      const { data } = await getJobJobsJobIdGet({ path: { job_id: jobId } });
      return data as Job;
    },
  });

  if (isLoading || !job) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
        <div className="text-[var(--accent)] animate-pulse text-lg">Loading...</div>
      </div>
    );
  }

  const events = job.result?.events || [];
  const filtered = events.filter(
    e =>
      (dirFilter === 'ALL' || e.direction === dirFilter) &&
      (methodFilter === 'ALL' || e.method === methodFilter),
  );

  const behaviorSummary = job.result?.behavior_summary || {};
  const behaviorPieData = Object.entries(behaviorSummary)
    .map(([k, v]) => ({ name: k.replace('_detections', ''), value: v as number }))
    .filter(d => d.value > 0);

  const exportCSV = () => {
    const headers = ['frame', 'time_sec', 'track_id', 'direction', 'method', 'speed_px_sec', 'behavior', 'angle_deg'];
    const rows = events.map(e => [
      e.frame, e.timestamp_sec.toFixed(2), e.track_id, e.direction,
      e.method, e.speed_px_per_sec.toFixed(1),
      e.behavior_class || '', e.angle_deg?.toFixed(1) || '',
    ]);
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `events_${job.job_id}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <div className="bg-[var(--bg-card)] rounded-xl w-full max-w-6xl max-h-[92vh] overflow-y-auto border border-gray-700 shadow-2xl">

        {/* Header */}
        <div className="sticky top-0 bg-[var(--bg-card)] border-b border-gray-800 p-4 flex justify-between items-start z-10">
          <div>
            <h2 className="text-base font-bold text-[var(--text-primary)]">{job.filename}</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              {new Date(job.created_at).toLocaleString()} ·
              Approach {job.result?.approach_used} · {job.result?.tracker_used} ·
              <span className="text-[var(--color-in)] font-bold"> {job.result?.total_in} IN</span> /
              <span className="text-[var(--color-out)] font-bold"> {job.result?.total_out} OUT</span> ·
              {job.result?.fps_processed.toFixed(1)} fps · {job.result?.duration_sec.toFixed(1)}s
            </p>
          </div>
          <div className="flex gap-2 shrink-0 ml-4">
            <button onClick={exportCSV} className="flex items-center gap-1.5 text-xs bg-gray-800 hover:bg-gray-700 px-3 py-1.5 rounded-lg transition-colors">
              <Download size={13} /> CSV
            </button>
            <button onClick={onClose} className="p-1.5 hover:bg-gray-800 rounded-full transition-colors">
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="p-4 space-y-5">

          {/* Charts row */}
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
            {/* Behavior pie */}
            <div className="bg-[var(--bg-panel)] p-3 rounded-xl border border-gray-800">
              <div className="text-xs font-semibold mb-2 text-gray-300 uppercase tracking-wide">Behavior</div>
              <ResponsiveContainer width="100%" height={150}>
                <PieChart>
                  <Pie data={behaviorPieData} cx="50%" cy="50%" innerRadius={30} outerRadius={60} dataKey="value">
                    {behaviorPieData.map(entry => (
                      <Cell key={entry.name} fill={BEHAVIOR_COLORS[entry.name] || '#718096'} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1">
                {behaviorPieData.map(d => (
                  <span key={d.name} className="text-xs flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full inline-block" style={{ background: BEHAVIOR_COLORS[d.name] || '#718096' }} />
                    {d.name} <span className="text-gray-500">({d.value})</span>
                  </span>
                ))}
              </div>
            </div>

            {/* Speed histogram */}
            <div className="bg-[var(--bg-panel)] p-3 rounded-xl border border-gray-800">
              <div className="text-xs font-semibold mb-2 text-gray-300 uppercase tracking-wide">Speed (px/s)</div>
              <SpeedHistogram events={events} />
            </div>

            {/* Angle histogram */}
            <div className="bg-[var(--bg-panel)] p-3 rounded-xl border border-gray-800">
              <div className="text-xs font-semibold mb-2 text-gray-300 uppercase tracking-wide">Angle (Approach B)</div>
              <AngleHistogram events={events} />
            </div>

            {/* Cumulative timeline */}
            <div className="bg-[var(--bg-panel)] p-3 rounded-xl border border-gray-800">
              <div className="text-xs font-semibold mb-2 text-gray-300 uppercase tracking-wide">Cumulative Traffic</div>
              <CumulativeTimeline events={events} />
            </div>
          </div>

          {/* Event timeline */}
          <div className="bg-[var(--bg-panel)] rounded-xl border border-gray-800 overflow-hidden">
            <div className="p-3 flex flex-wrap gap-2 items-center border-b border-gray-800">
              <span className="text-xs font-semibold text-gray-300 uppercase tracking-wide">
                Events <span className="text-gray-500 font-normal">({filtered.length} / {events.length})</span>
              </span>
              <div className="flex gap-1 ml-auto">
                {(['ALL', 'IN', 'OUT'] as const).map(f => (
                  <button
                    key={f}
                    onClick={() => setDirFilter(f)}
                    className={`text-xs px-2 py-1 rounded transition-colors ${dirFilter === f ? 'bg-[var(--accent)] text-black font-bold' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
                  >{f}</button>
                ))}
              </div>
              <div className="flex gap-1">
                {[
                  { key: 'ALL', label: 'All' },
                  { key: 'pose_confirmed', label: 'Pose ✓' },
                  { key: 'trajectory_fallback', label: 'Fallback' },
                ].map(({ key, label }) => (
                  <button
                    key={key}
                    onClick={() => setMethodFilter(key)}
                    className={`text-xs px-2 py-1 rounded transition-colors ${methodFilter === key ? 'bg-blue-700 text-white font-bold' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
                  >{label}</button>
                ))}
              </div>
            </div>

            <div className="overflow-y-auto max-h-72">
              <table className="w-full text-xs text-left text-gray-400">
                <thead className="sticky top-0 bg-gray-800/90 text-gray-500 uppercase">
                  <tr>
                    <th className="px-3 py-2">Time</th>
                    <th className="px-3 py-2">Track</th>
                    <th className="px-3 py-2">Dir</th>
                    <th className="px-3 py-2">Method</th>
                    <th className="px-3 py-2">Speed</th>
                    <th className="px-3 py-2">Behavior</th>
                    <th className="px-3 py-2">Angle</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((e, i) => (
                    <tr key={i} className="border-b border-gray-800/40 hover:bg-gray-800/30 transition-colors">
                      <td className="px-3 py-1.5 font-mono">{e.timestamp_sec.toFixed(2)}s</td>
                      <td className="px-3 py-1.5 font-mono text-gray-500">#{e.track_id}</td>
                      <td className="px-3 py-1.5">
                        <span className={`font-bold ${e.direction === 'IN' ? 'text-[var(--color-in)]' : 'text-[var(--color-out)]'}`}>
                          {e.direction}
                        </span>
                      </td>
                      <td className="px-3 py-1.5">
                        <span className={`px-1.5 py-0.5 rounded text-xs ${e.method === 'pose_confirmed' ? 'bg-blue-900/60 text-blue-400' : 'bg-gray-800 text-gray-500'}`}>
                          {e.method === 'pose_confirmed' ? 'pose ✓' : e.method === 'trajectory_fallback' ? 'fallback' : 'traj'}
                        </span>
                      </td>
                      <td className="px-3 py-1.5 font-mono">{e.speed_px_per_sec.toFixed(0)}</td>
                      <td className="px-3 py-1.5">
                        {e.behavior_class ? (
                          <span
                            className="px-1.5 py-0.5 rounded text-xs"
                            style={{ background: (BEHAVIOR_COLORS[e.behavior_class] || '#718096') + '28', color: BEHAVIOR_COLORS[e.behavior_class] || '#718096' }}
                          >
                            {e.behavior_class}
                          </span>
                        ) : <span className="text-gray-700">—</span>}
                      </td>
                      <td className="px-3 py-1.5 font-mono text-gray-500">
                        {e.angle_deg !== null ? `${e.angle_deg.toFixed(1)}°` : '—'}
                      </td>
                    </tr>
                  ))}
                  {filtered.length === 0 && (
                    <tr>
                      <td colSpan={7} className="px-3 py-8 text-center text-gray-600">Немає подій за фільтром</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
