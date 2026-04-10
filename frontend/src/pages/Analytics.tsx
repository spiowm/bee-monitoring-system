import { useState, useEffect } from 'react';
import { API } from '../api/client';
import { BarChart, Activity, Layers, ActivitySquare, CheckCircle, History, Trash2, Play, Download, X } from 'lucide-react';

export default function AnalyticsPage() {
  const [summary, setSummary] = useState<any>(null);
  const [compare, setCompare] = useState<any>(null);
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [playingVideo, setPlayingVideo] = useState<string | null>(null);

  const handleDelete = async (jobId: string) => {
    if (!confirm('Are you sure you want to delete this job and its video?')) return;
    try {
      await API.deleteJob(jobId);
      setJobs(prev => prev.filter(j => j.job_id !== jobId));
    } catch (e) {
      console.error(e);
      alert('Failed to delete job');
    }
  };

  useEffect(() => {
    async function load() {
      try {
        const [sum, comp, jbs] = await Promise.all([
          API.getAnalyticsSummary(),
          API.getApproachCompare(),
          API.listJobs()
        ]);
        setSummary(sum);
        setCompare(comp);
        setJobs(jbs);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <div className="animate-pulse text-center p-12 text-[var(--accent)]"><Activity className="mx-auto block" size={48} />Loading Analytics...</div>;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold border-b border-gray-800 pb-2 flex items-center gap-2">
        <BarChart className="text-[var(--accent)]" /> Diploma Research Analytics
      </h1>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card flex items-center gap-4">
          <div className="p-3 bg-gray-800 rounded-lg text-gray-300"><Layers size={24} /></div>
          <div>
            <div className="text-sm text-gray-400">Total Sessions (Jobs)</div>
            <div className="text-xl font-bold">{summary?.total_sessions || 0}</div>
          </div>
        </div>
        <div className="card flex items-center gap-4">
          <div className="p-3 bg-[var(--color-in)]/20 rounded-lg text-[var(--color-in)]"><ActivitySquare size={24} /></div>
          <div>
            <div className="text-sm text-gray-400">Total IN Events</div>
            <div className="text-xl font-bold">{summary?.total_in || 0}</div>
          </div>
        </div>
        <div className="card flex items-center gap-4">
          <div className="p-3 bg-[var(--color-out)]/20 rounded-lg text-[var(--color-out)]"><ActivitySquare size={24} /></div>
          <div>
            <div className="text-sm text-gray-400">Total OUT Events</div>
            <div className="text-xl font-bold">{summary?.total_out || 0}</div>
          </div>
        </div>
        <div className="card flex items-center gap-4">
          <div className="p-3 bg-blue-900/50 rounded-lg text-blue-400"><CheckCircle size={24} /></div>
          <div>
            <div className="text-sm text-gray-400">Avg Balance (IN - OUT)</div>
            <div className="text-xl font-bold">{(summary?.avg_balance || 0).toFixed(1)}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
         {/* Approach Comparison */}
         <div className="card col-span-1 lg:col-span-2 space-y-4 border border-[var(--accent)] shadow-[0_0_15px_rgba(240,180,41,0.1)]">
            <h2 className="text-lg font-bold flex items-center gap-2"><Activity size={20} className="text-[var(--accent)]" /> Approach Comparison: A vs B</h2>
            <p className="text-sm text-gray-400">Порівняння підрахунку тільки на основі траєкторії (A) проти фільтрації за pose векторами (B).</p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4">
               <div className="bg-[var(--bg-panel)] p-4 rounded-xl border border-gray-700">
                  <div className="font-semibold text-gray-300 mb-2 border-b border-gray-800 pb-2">Approach A (Trajectory)</div>
                  <div className="text-3xl font-bold font-mono">{compare?.avg_in_a?.toFixed(1) || 0}</div>
                  <div className="text-xs text-gray-500 mt-1">Avg IN per session ({compare?.approach_a_count || 0} samples)</div>
               </div>
               
               <div className="bg-[var(--bg-panel)] p-4 rounded-xl border border-[var(--accent)]/50 relative overflow-hidden">
                  <div className="absolute top-0 right-0 bg-[var(--accent)] text-xs text-black font-bold px-2 py-1 rounded-bl-lg">PROPOSED</div>
                  <div className="font-semibold text-[var(--accent)] mb-2 border-b border-gray-800 pb-2">Approach B (Pose Filter)</div>
                  <div className="text-3xl font-bold font-mono">{compare?.avg_in_b?.toFixed(1) || 0}</div>
                  <div className="text-xs text-xl mt-1 text-gray-400">Avg IN per session ({compare?.approach_b_count || 0} samples)</div>
               </div>
               
               <div className="bg-[var(--bg-panel)] p-4 rounded-xl border border-blue-900/50">
                  <div className="font-semibold text-blue-400 mb-2 border-b border-gray-800 pb-2">Pose Confirmation Rate</div>
                  <div className="text-3xl font-bold font-mono text-blue-400">{compare?.pose_confirmed_rate?.toFixed(1) || 0}%</div>
                  <div className="text-xs text-gray-500 mt-1">of total events validated by Pose</div>
               </div>
            </div>
            
            <div className="bg-gray-800/50 p-4 rounded-lg mt-4 text-sm text-gray-300 leading-relaxed border-l-4 border-[var(--accent)]">
               <strong className="text-[var(--text-primary)]">Висновок для дослідження: </strong> 
               Підхід B (оснований на виділенні вектора голова-жало) показує рівень підтвердження {compare?.pose_confirmed_rate?.toFixed(1) || 0}%. 
               Використання fallaback механізму забезпечує стабільність підрахунку в моменти коли keypoints недоступні. 
            </div>
         </div>

         {/* Job History */}
         <div className="card col-span-1 lg:col-span-2 space-y-4">
            <h2 className="text-lg font-bold flex items-center gap-2 border-b border-gray-800 pb-2"><History size={20} className="text-[var(--accent)]" /> 
              Recent Job History
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left text-gray-400">
                <thead className="text-xs text-gray-500 uppercase bg-gray-800/50">
                  <tr>
                    <th className="px-4 py-3 rounded-tl-lg">Date</th>
                    <th className="px-4 py-3">File</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Config</th>
                    <th className="px-4 py-3">Traffic (IN/OUT)</th>
                    <th className="px-4 py-3">Duration</th>
                    <th className="px-4 py-3 rounded-tr-lg">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map(j => (
                    <tr key={j.job_id} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                      <td className="px-4 py-3 whitespace-nowrap">{new Date(j.created_at).toLocaleString()}</td>
                      <td className="px-4 py-3 font-medium text-gray-300 max-w-[200px] truncate" title={j.filename}>{j.filename}</td>
                      <td className="px-4 py-3 whitespace-nowrap">
                         <span className={`px-2 py-1 rounded-full text-xs font-medium ${j.status === 'complete' ? 'bg-green-500/20 text-green-400 border border-green-500/30' : j.status === 'failed' ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'}`}>
                           {j.status}
                         </span>
                      </td>
                      <td className="px-4 py-3 text-xs opacity-80 whitespace-nowrap">
                         Appr: {j.config?.approach}, Trk: {j.config?.tracker_name}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                         {j.result ? <span className="font-mono font-bold text-green-400">{j.result.total_in}</span> : '-'} / {j.result ? <span className="font-mono font-bold text-red-400">{j.result.total_out}</span> : '-'}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">{j.result?.duration_sec?.toFixed(1) ? `${j.result.duration_sec.toFixed(1)}s` : '-'}</td>
                      <td className="px-4 py-3 flex gap-3 items-center whitespace-nowrap">
                        {j.result?.annotated_video_url && (
                          <>
                            <button onClick={() => setPlayingVideo(`http://localhost:8000${j.result?.annotated_video_url}`)} className="text-blue-400 hover:text-blue-300 transition-colors" title="Play">
                              <Play size={16} />
                            </button>
                            <a href={`http://localhost:8000${j.result?.annotated_video_url}`} download className="text-green-400 hover:text-green-300 transition-colors" title="Download">
                              <Download size={16} />
                            </a>
                          </>
                        )}
                        <button onClick={() => handleDelete(j.job_id)} className="text-red-400 hover:text-red-300 transition-colors" title="Delete">
                           <Trash2 size={16} />
                        </button>
                      </td>
                    </tr>
                  ))}
                  {jobs.length === 0 && (
                    <tr><td colSpan={7} className="px-4 py-8 text-center text-gray-500">No jobs found yet.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
         </div>
      </div>
      
      {/* Video Player Modal */}
      {playingVideo && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-[var(--bg-panel)] rounded-xl overflow-hidden w-full max-w-4xl relative shadow-2xl border border-gray-700">
            <div className="absolute top-4 right-4 z-10">
              <button onClick={() => setPlayingVideo(null)} className="p-2 bg-gray-900/80 hover:bg-gray-800 rounded-full text-gray-300 hover:text-white transition-colors">
                <X size={20} />
              </button>
            </div>
            <video src={playingVideo} controls autoPlay className="w-full max-h-[85vh] bg-black" />
          </div>
        </div>
      )}
    </div>
  );
}
