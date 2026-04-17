import type { LiveStats, Job } from '../types';

interface LiveStatsPanelProps {
  liveStats: LiveStats | null;
  job: Job | null;
}

export default function LiveStatsPanel({ liveStats, job }: LiveStatsPanelProps) {
  return (
    <div className="card h-full flex flex-col">
      <h2 className="text-lg font-semibold border-b border-gray-700 pb-2 mb-4">Traffic Live Stats</h2>
      
      <div className="grid grid-cols-2 gap-3 mb-6">
        <div className="bg-[var(--bg-panel)] p-3 rounded-lg border border-[var(--color-in)]/20 shadow-[0_0_10px_rgba(72,187,120,0.1)]">
          <div className="text-xs text-gray-400 mb-1">IN Traffic</div>
          <div className="text-2xl font-bold text-[var(--color-in)]">{liveStats?.total_in || job?.result?.total_in || 0}</div>
        </div>
        <div className="bg-[var(--bg-panel)] p-3 rounded-lg border border-[var(--color-out)]/20 shadow-[0_0_10px_rgba(252,129,129,0.1)]">
          <div className="text-xs text-gray-400 mb-1">OUT Traffic</div>
          <div className="text-2xl font-bold text-[var(--color-out)]">{liveStats?.total_out || job?.result?.total_out || 0}</div>
        </div>
        <div className="col-span-2 bg-[var(--bg-panel)] p-3 rounded-lg flex justify-between items-center border border-gray-700">
           <span className="text-sm font-medium text-gray-300">Active Bees on Ramp</span>
           <span className="text-xl font-bold text-[var(--accent)]">{liveStats?.bees_on_ramp || 0}</span>
        </div>
      </div>

      <div className="space-y-2 mb-6 text-sm flex-grow">
         <div className="flex justify-between border-b border-gray-800 pb-1">
           <span className="text-gray-400">Pipeline Speed</span>
           <span className="font-mono">{liveStats?.current_fps?.toFixed(1) || job?.result?.fps_processed?.toFixed(1) || 0} FPS</span>
         </div>
         <div className="flex justify-between border-b border-gray-800 pb-1">
           <span className="text-gray-400">Pose Matches</span>
           <span className="text-[var(--color-pose)]">{liveStats?.pose_confirmed || job?.result?.pose_confirmed_events || 0}</span>
         </div>
         <div className="flex justify-between border-b border-gray-800 pb-1">
           <span className="text-gray-400">Fallback Tracker</span>
           <span className="text-[var(--color-fallback)]">{liveStats?.fallback_events || job?.result?.fallback_events || 0}</span>
         </div>
      </div>

      <div>
         <h3 className="text-sm font-semibold text-gray-300 mb-2">Bee Behavior (Heuristics)</h3>
         <div className="space-y-1 text-xs">
            {['foraging', 'fanning', 'guarding', 'washboarding'].map(b => (
               <div key={b} className="flex justify-between items-center p-1.5 bg-[var(--bg-panel)] rounded">
                 <span className="capitalize text-gray-400">{b}</span>
                 <span className="font-mono font-medium">{liveStats?.behavior_counts?.[b] || job?.result?.behavior_summary?.[`${b}_detections`] || 0}</span>
               </div>
            ))}
         </div>
      </div>
    </div>
  );
}
