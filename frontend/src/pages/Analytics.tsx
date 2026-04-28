import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getSummaryAnalyticsSummaryGet,
  compareApproachesAnalyticsCompareApproachesGet,
  listJobsJobsGet,
  deleteJobJobsJobIdDelete,
} from '../api/generated';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts';
import {
  BarChart2, Activity, Layers, ActivitySquare, CheckCircle,
  History, Trash2, Play, Download, X, Info,
} from 'lucide-react';
import type { Job } from '../types';
import JobDetailModal from '../components/JobDetailModal';
import RunComparisonModal from '../components/RunComparisonModal';

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

export default function AnalyticsPage() {
  const [playingVideo, setPlayingVideo] = useState<string | null>(null);
  const [detailJobId, setDetailJobId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showComparison, setShowComparison] = useState(false);

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.size < 4 && next.add(id);
      return next;
    });
  };
  const queryClient = useQueryClient();

  const { data: summary, isLoading: loadingSummary } = useQuery({
    queryKey: ['analyticsSummary'],
    queryFn: async () => {
      const { data } = await getSummaryAnalyticsSummaryGet();
      return data as any;
    },
  });

  const { data: compare, isLoading: loadingCompare } = useQuery({
    queryKey: ['analyticsCompare'],
    queryFn: async () => {
      const { data } = await compareApproachesAnalyticsCompareApproachesGet();
      return data as any;
    },
  });

  const { data: jobs = [], isLoading: loadingJobs } = useQuery({
    queryKey: ['jobs'],
    queryFn: async () => {
      const { data } = await listJobsJobsGet();
      return data as Job[];
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (jobId: string) => {
      await deleteJobJobsJobIdDelete({ path: { job_id: jobId } });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      queryClient.invalidateQueries({ queryKey: ['analyticsSummary'] });
      queryClient.invalidateQueries({ queryKey: ['analyticsCompare'] });
    },
  });

  const completedJobs = jobs.filter(j => j.status === 'complete' && j.result);

  // Aggregate behavior across all completed jobs
  const totalBehavior: Record<string, number> = {};
  completedJobs.forEach(j => {
    Object.entries(j.result!.behavior_summary).forEach(([k, v]) => {
      const name = k.replace('_detections', '');
      totalBehavior[name] = (totalBehavior[name] || 0) + (v as number);
    });
  });
  const behaviorPieData = Object.entries(totalBehavior)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => ({ name: k, value: v }));

  // Sessions bar chart (last 10 completed jobs)
  const sessionsData = completedJobs
    .slice(0, 10)
    .reverse()
    .map(j => ({
      name: new Date(j.created_at).toLocaleDateString('uk', { month: 'short', day: 'numeric' }),
      IN: j.result!.total_in,
      OUT: j.result!.total_out,
      fps: parseFloat(j.result!.fps_processed.toFixed(1)),
    }));

  const loading = loadingSummary || loadingCompare || loadingJobs;
  if (loading) {
    return (
      <div className="animate-pulse text-center p-12 text-[var(--accent)]">
        <Activity className="mx-auto block" size={48} />
        <p className="mt-2">Loading Analytics...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold border-b border-gray-800 pb-2 flex items-center gap-2">
        <BarChart2 className="text-[var(--accent)]" /> Analytics & Research
      </h1>

      {/* Overview Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card flex items-center gap-4">
          <div className="p-3 bg-gray-800 rounded-lg text-gray-300"><Layers size={22} /></div>
          <div>
            <div className="text-xs text-gray-400">Sessions</div>
            <div className="text-2xl font-bold">{summary?.total_sessions || 0}</div>
          </div>
        </div>
        <div className="card flex items-center gap-4">
          <div className="p-3 bg-[var(--color-in)]/20 rounded-lg text-[var(--color-in)]"><ActivitySquare size={22} /></div>
          <div>
            <div className="text-xs text-gray-400">Total IN</div>
            <div className="text-2xl font-bold text-[var(--color-in)]">{summary?.total_in || 0}</div>
          </div>
        </div>
        <div className="card flex items-center gap-4">
          <div className="p-3 bg-[var(--color-out)]/20 rounded-lg text-[var(--color-out)]"><ActivitySquare size={22} /></div>
          <div>
            <div className="text-xs text-gray-400">Total OUT</div>
            <div className="text-2xl font-bold text-[var(--color-out)]">{summary?.total_out || 0}</div>
          </div>
        </div>
        <div className="card flex items-center gap-4">
          <div className="p-3 bg-blue-900/50 rounded-lg text-blue-400"><CheckCircle size={22} /></div>
          <div>
            <div className="text-xs text-gray-400">Avg Balance</div>
            <div className="text-2xl font-bold">{(summary?.avg_balance || 0).toFixed(1)}</div>
          </div>
        </div>
      </div>

      {/* Aggregate charts */}
      {completedJobs.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Behavior distribution */}
          <div className="card space-y-2">
            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
              Загальний розподіл поведінки ({completedJobs.length} сесій)
            </h2>
            <div className="flex items-center gap-4">
              <ResponsiveContainer width={160} height={160}>
                <PieChart>
                  <Pie data={behaviorPieData} cx="50%" cy="50%" innerRadius={35} outerRadius={70} dataKey="value">
                    {behaviorPieData.map(entry => (
                      <Cell key={entry.name} fill={BEHAVIOR_COLORS[entry.name] || '#718096'} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-2">
                {behaviorPieData.map(d => (
                  <div key={d.name} className="flex items-center gap-2 text-sm">
                    <span className="w-3 h-3 rounded-full shrink-0" style={{ background: BEHAVIOR_COLORS[d.name] || '#718096' }} />
                    <span className="text-gray-300 capitalize">{d.name}</span>
                    <span className="text-gray-500 ml-auto">{d.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sessions traffic chart */}
          <div className="card space-y-2">
            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
              Трафік по сесіях
            </h2>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={sessionsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="IN" fill="var(--color-in)" radius={[2, 2, 0, 0]} />
                <Bar dataKey="OUT" fill="var(--color-out)" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Approach Comparison */}
      <div className="card space-y-4 border border-[var(--accent)] shadow-[0_0_15px_rgba(240,180,41,0.08)]">
        <h2 className="text-base font-bold flex items-center gap-2">
          <Activity size={18} className="text-[var(--accent)]" /> Approach A vs B
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-[var(--bg-panel)] p-4 rounded-xl border border-gray-700">
            <div className="text-xs font-semibold text-gray-400 mb-2 uppercase">Approach A (Trajectory)</div>
            <div className="text-3xl font-bold font-mono">{compare?.avg_in_a?.toFixed(1) || 0}</div>
            <div className="text-xs text-gray-500 mt-1">avg IN / session ({compare?.approach_a_count || 0} samples)</div>
          </div>
          <div className="bg-[var(--bg-panel)] p-4 rounded-xl border border-[var(--accent)]/50 relative overflow-hidden">
            <div className="absolute top-0 right-0 bg-[var(--accent)] text-xs text-black font-bold px-2 py-1 rounded-bl-lg">PROPOSED</div>
            <div className="text-xs font-semibold text-[var(--accent)] mb-2 uppercase">Approach B (Pose Filter)</div>
            <div className="text-3xl font-bold font-mono">{compare?.avg_in_b?.toFixed(1) || 0}</div>
            <div className="text-xs text-gray-400 mt-1">avg IN / session ({compare?.approach_b_count || 0} samples)</div>
          </div>
          <div className="bg-[var(--bg-panel)] p-4 rounded-xl border border-blue-900/50">
            <div className="text-xs font-semibold text-blue-400 mb-2 uppercase">Pose Confirmation Rate</div>
            <div className="text-3xl font-bold font-mono text-blue-400">{compare?.pose_confirmed_rate?.toFixed(1) || 0}%</div>
            <div className="text-xs text-gray-500 mt-1">events validated by pose vector</div>
          </div>
        </div>
        <div className="bg-gray-800/40 p-3 rounded-lg text-sm text-gray-300 border-l-4 border-[var(--accent)]">
          <strong>Висновок: </strong>
          Підхід B підтверджує {compare?.pose_confirmed_rate?.toFixed(1) || 0}% подій pose-вектором голова→жало.
          Fallback забезпечує стабільність при відсутності keypoints.
        </div>
      </div>

      {/* Job History */}
      <div className="card space-y-4">
        <div className="flex items-center justify-between border-b border-gray-800 pb-2">
          <h2 className="text-base font-bold flex items-center gap-2">
            <History size={18} className="text-[var(--accent)]" /> Job History
          </h2>
          {selectedIds.size >= 2 && (
            <button
              onClick={() => setShowComparison(true)}
              className="text-xs bg-[var(--accent)] text-black font-bold px-3 py-1.5 rounded-lg hover:opacity-90 transition-opacity"
            >
              Compare {selectedIds.size} runs
            </button>
          )}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left text-gray-400">
            <thead className="text-xs text-gray-500 uppercase bg-gray-800/50">
              <tr>
                <th className="px-3 py-3 w-8"></th>
                <th className="px-4 py-3">Date</th>
                <th className="px-4 py-3">File</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Config</th>
                <th className="px-4 py-3">Traffic</th>
                <th className="px-4 py-3">FPS</th>
                <th className="px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map(j => (
                <tr key={j.job_id} className={`border-b border-gray-800/50 hover:bg-gray-800/20 transition-colors ${selectedIds.has(j.job_id) ? 'bg-[var(--accent)]/5' : ''}`}>
                  <td className="px-3 py-3">
                    {j.status === 'complete' && j.result && (
                      <input
                        type="checkbox"
                        checked={selectedIds.has(j.job_id)}
                        onChange={() => toggleSelect(j.job_id)}
                        className="accent-[var(--accent)] w-3.5 h-3.5 cursor-pointer"
                        title={selectedIds.size >= 4 && !selectedIds.has(j.job_id) ? 'Max 4 runs' : ''}
                        disabled={selectedIds.size >= 4 && !selectedIds.has(j.job_id)}
                      />
                    )}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-xs">
                    {new Date(j.created_at).toLocaleString('uk')}
                  </td>
                  <td className="px-4 py-3 font-medium text-gray-300 max-w-[180px] truncate text-xs" title={j.filename}>
                    {j.filename}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                      j.status === 'complete' ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                      : j.status === 'failed' ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                      : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                    }`}>{j.status}</span>
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-500 whitespace-nowrap">
                    {j.config?.approach} · {j.config?.tracker_name}
                    {(j.config as any)?.model_name && (
                      <span className="ml-1 text-[var(--accent)]">· {(j.config as any).model_name}</span>
                    )}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap font-mono text-sm">
                    {j.result ? (
                      <>
                        <span className="text-[var(--color-in)] font-bold">{j.result.total_in}</span>
                        <span className="text-gray-600"> / </span>
                        <span className="text-[var(--color-out)] font-bold">{j.result.total_out}</span>
                      </>
                    ) : '—'}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-xs">
                    {j.result?.fps_processed?.toFixed(1) || '—'}
                  </td>
                  <td className="px-4 py-3 flex gap-2 items-center whitespace-nowrap">
                    {j.status === 'complete' && j.result && (
                      <button
                        onClick={() => setDetailJobId(j.job_id)}
                        className="text-[var(--accent)] hover:text-yellow-300 transition-colors"
                        title="Details"
                      >
                        <Info size={15} />
                      </button>
                    )}
                    {j.result?.annotated_video_url && (
                      <>
                        <button
                          onClick={() => setPlayingVideo(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${j.result?.annotated_video_url}`)}
                          className="text-blue-400 hover:text-blue-300 transition-colors"
                          title="Play"
                        >
                          <Play size={15} />
                        </button>
                        <a
                          href={`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${j.result?.annotated_video_url}`}
                          download
                          className="text-green-400 hover:text-green-300 transition-colors"
                          title="Download"
                        >
                          <Download size={15} />
                        </a>
                      </>
                    )}
                    <button
                      onClick={() => {
                        if (!confirm('Видалити цей job?')) return;
                        deleteMutation.mutate(j.job_id);
                      }}
                      disabled={deleteMutation.isPending}
                      className="text-red-400 hover:text-red-300 transition-colors disabled:opacity-50"
                      title="Delete"
                    >
                      <Trash2 size={15} />
                    </button>
                  </td>
                </tr>
              ))}
              {jobs.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-4 py-8 text-center text-gray-500">No jobs yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Video Modal */}
      {playingVideo && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="bg-[var(--bg-panel)] rounded-xl overflow-hidden w-full max-w-4xl relative shadow-2xl border border-gray-700">
            <button
              onClick={() => setPlayingVideo(null)}
              className="absolute top-3 right-3 z-10 p-2 bg-gray-900/80 hover:bg-gray-800 rounded-full text-gray-300 hover:text-white transition-colors"
            >
              <X size={18} />
            </button>
            <video src={playingVideo} controls autoPlay className="w-full max-h-[85vh] bg-black" />
          </div>
        </div>
      )}

      {/* Job Detail Modal */}
      {detailJobId && (
        <JobDetailModal jobId={detailJobId} onClose={() => setDetailJobId(null)} />
      )}

      {/* Run Comparison Modal */}
      {showComparison && selectedIds.size >= 2 && (
        <RunComparisonModal
          jobs={jobs.filter(j => selectedIds.has(j.job_id))}
          onClose={() => setShowComparison(false)}
        />
      )}
    </div>
  );
}
