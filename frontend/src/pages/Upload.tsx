import { useState, useRef, useEffect } from 'react';
import { API } from '../api/client';
import type { ProcessConfig, VizConfig, Job, LiveStats } from '../api/client';
import { Settings, Sliders, Play, Download, AlertCircle, ChevronDown, ChevronRight, Video, UploadCloud, Activity } from 'lucide-react';

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [liveStats, setLiveStats] = useState<LiveStats | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [testVideos, setTestVideos] = useState<string[]>([]);

  // Poll interval ref
  const pollRef = useRef<number | null>(null);

  const [config, setConfig] = useState<ProcessConfig>({
    tracker_name: 'bytetrack',
    approach: 'A',
    line_position: 0.5,
    conf_threshold: 0.35,
    kp_conf_threshold: 0.5,
    track_tail_length: 30,
    angle_threshold_deg: 60.0,
    ramp_detect_interval: 30
  });

  const [vizConfig, setVizConfig] = useState<VizConfig>({
    show_boxes: true, show_ids: true, show_confidence: true,
    show_keypoints: true, show_ramp: true, show_behaviors: true,
    show_counting_line: true, show_stats_overlay: true,
    show_tracks: true, show_orientation: true, show_recent_events: true
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showViz, setShowViz] = useState(false);

  useEffect(() => {
    if (jobId && isProcessing) {
      pollRef.current = window.setInterval(async () => {
        try {
          const data = await API.getJobLiveStats(jobId);
          setLiveStats(data.live_stats);
          if (data.status === 'complete' || data.status === 'failed') {
            setIsProcessing(false);
            if (pollRef.current) clearInterval(pollRef.current);
            const fullJob = await API.getJob(jobId);
            setJob(fullJob);
          }
        } catch (e) {
          console.error(e);
        }
      }, 2000);
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId, isProcessing]);

  useEffect(() => {
    API.getTestVideos().then(setTestVideos).catch(console.error);
  }, []);

  const handleStart = async () => {
    if (!file) return;
    try {
      const res = await API.createJob(file, config, vizConfig);
      setJobId(res.job_id);
      setIsProcessing(true);
      setJob(null);
      setLiveStats(null);
    } catch (e) {
      console.error(e);
      alert('Failed to start Job');
    }
  };

  const handleStartTest = async (filename: string) => {
    try {
      const res = await API.createTestJob(filename, config, vizConfig);
      setJobId(res.job_id);
      setIsProcessing(true);
      setJob(null);
      setLiveStats(null);
      setFile(null); // Clear custom uploaded file
    } catch (e) {
      console.error(e);
      alert('Failed to start Test Job');
    }
  };

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-full flex-grow">
      
      {/* 1. LEFT COLUMN: Config */}
      <div className="w-full lg:w-[320px] shrink-0 space-y-4">
        <div className="card space-y-4">
          <h2 className="text-lg font-semibold border-b border-gray-700 pb-2 mb-4 flex items-center gap-2">
            <UploadCloud size={20} className="text-[var(--accent)]" /> Upload Video
          </h2>
          <div className="border-2 border-dashed border-gray-700 hover:border-[var(--accent)] cursor-pointer rounded-xl p-6 text-center transition-colors bg-[var(--bg-panel)] relative">
            <input type="file" accept="video/*" onChange={e => setFile(e.target.files?.[0] || null)} className="absolute inset-0 opacity-0 cursor-pointer" />
            {!file ? (
              <div className="text-gray-400 text-sm">
                <Video size={36} className="mx-auto mb-2 opacity-50" />
                <p>Drag & drop or click</p>
                <p className="text-xs mt-1 opacity-70">MP4, AVI, MOV • Max 500MB</p>
              </div>
            ) : (
              <div className="text-green-400 font-medium break-all">
                {file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)
              </div>
            )}
          </div>

          <div className="flex flex-wrap gap-2 mt-2">
            {testVideos.map(tv => (
              <button 
                 key={tv}
                 onClick={() => handleStartTest(tv)} 
                 disabled={isProcessing}
                 className="bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs py-2 px-3 rounded-lg flex-1 min-w-[120px] transition-colors"
              >
                 Run `{tv}`
              </button>
            ))}
            {testVideos.length === 0 && <span className="text-xs text-gray-500">No test videos found</span>}
          </div>

          <div className="pt-2">
            <label className="text-sm text-gray-400 mb-1 block">Tracker</label>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 text-sm"><input type="radio" checked={config.tracker_name === 'bytetrack'} onChange={() => setConfig({...config, tracker_name: 'bytetrack'})} className="accent-[var(--accent)]" /> ByteTrack</label>
              <label className="flex items-center gap-2 text-sm"><input type="radio" checked={config.tracker_name === 'ocsort'} onChange={() => setConfig({...config, tracker_name: 'ocsort'})} className="accent-[var(--accent)]" /> OC-SORT</label>
            </div>
          </div>

          <div className="pt-2">
            <label className="text-sm text-gray-400 mb-1 block">Counting Approach</label>
            <div className="flex flex-col gap-2">
              <label className="flex items-center gap-2 text-sm"><input type="radio" checked={config.approach === 'A'} onChange={() => setConfig({...config, approach: 'A'})} className="accent-[var(--accent)]" /> A: Trajectory Only</label>
              <label className="flex items-center gap-2 text-sm"><input type="radio" checked={config.approach === 'B'} onChange={() => setConfig({...config, approach: 'B'})} className="accent-[var(--accent)]" /> B: Pose Filtered</label>
            </div>
          </div>

          <div className="pt-2">
            <label className="text-sm text-gray-400 flex justify-between">
              Line Position <span>{config.line_position.toFixed(2)}</span>
            </label>
            <input type="range" min="0.1" max="0.9" step="0.05" value={config.line_position} onChange={e => setConfig({...config, line_position: parseFloat(e.target.value)})} className="w-full accent-[var(--accent)]" />
          </div>
          
          <button className="flex items-center justify-between w-full text-sm font-semibold text-gray-300 py-2 border-t border-gray-700 mt-4" onClick={() => setShowAdvanced(!showAdvanced)}>
            <span className="flex items-center gap-2"><Settings size={16} /> Advanced Settings</span>
            {showAdvanced ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </button>
          {showAdvanced && (
            <div className="space-y-3 bg-[var(--bg-panel)] p-3 rounded-lg border border-gray-800">
               <label className="text-xs text-gray-400 block">Angle Threshold (°)</label>
               <input type="range" min="30" max="90" step="5" value={config.angle_threshold_deg} onChange={e => setConfig({...config, angle_threshold_deg: parseFloat(e.target.value)})} className="w-full accent-[var(--accent)] h-1" />
               <div className="text-xs text-right text-gray-300">{config.angle_threshold_deg}°</div>
            </div>
          )}

          <button className="flex items-center justify-between w-full text-sm font-semibold text-gray-300 py-2 border-t border-gray-700" onClick={() => setShowViz(!showViz)}>
             <span className="flex items-center gap-2"><Sliders size={16} /> Visualization Config</span>
             {showViz ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </button>
          {showViz && (
            <div className="grid grid-cols-2 gap-2 bg-[var(--bg-panel)] p-3 rounded-lg border border-gray-800 text-xs">
              {Object.keys(vizConfig).map(k => (
                <label key={k} className="flex items-center gap-1.5 cursor-pointer">
                  <input type="checkbox" checked={(vizConfig as any)[k]} onChange={e => setVizConfig({...vizConfig, [k]: e.target.checked})} className="accent-[var(--accent)] rounded" />
                  <span className="truncate" title={k}>{k.replace('show_', '')}</span>
                </label>
              ))}
            </div>
          )}

          <button 
            disabled={(!file && !jobId) && isProcessing} 
            onClick={handleStart}
            className={`btn-primary w-full flex justify-center items-center gap-2 mt-4 ${(!file && !jobId) || isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}>
            {isProcessing ? <span className="animate-pulse">Processing...</span> : <><Play size={18} /> Start Pipeline with File</>}
          </button>
        </div>
      </div>

      {/* 2. CENTER COLUMN: Video Player */}
      <div className="flex-grow flex flex-col gap-4 min-w-0">
        <div className="card flex-grow flex flex-col items-center justify-center p-0 overflow-hidden relative min-h-[400px]">
          {!job && !isProcessing && (
            <div className="text-center text-gray-500">
              <Video size={48} className="mx-auto mb-4 opacity-30" />
              <p>Select a video and press Start Pipeline</p>
            </div>
          )}
          
          {isProcessing && (
            <div className="w-full max-w-md p-6 text-center">
              <div className="text-[var(--accent)] mb-4 animate-bounce"><Activity size={48} className="mx-auto" /></div>
              <h3 className="text-xl font-bold mb-2">Analyzing Video...</h3>
              <div className="w-full bg-gray-800 rounded-full h-3 mb-2 overflow-hidden">
                <div className="bg-[var(--accent)] h-3 rounded-full transition-all duration-300" style={{width: `${(liveStats?.current_frame || 0) / Math.max(1, (liveStats?.total_frames || 1)) * 100}%`}}></div>
              </div>
              <p className="text-sm text-gray-400">Frame {liveStats?.current_frame || 0} / {liveStats?.total_frames || '?'}</p>
            </div>
          )}

          {job && job.status === 'complete' && job.result && (
            <div className="w-full h-full flex flex-col">
              <div className="bg-black flex-grow flex items-center justify-center relative">
                 <video src={`http://localhost:8000${job.result.annotated_video_url}`} controls className="max-h-full max-w-full" />
              </div>
              <div className="p-4 bg-[var(--bg-panel)] flex justify-between items-center border-t border-gray-800">
                <div>
                   <span className="font-bold text-[var(--accent)] text-lg">Analysis Complete</span>
                   <p className="text-xs text-gray-400">Processed in {job.result.duration_sec.toFixed(1)}s ({job.result.fps_processed.toFixed(1)} fps)</p>
                </div>
                <a href={`http://localhost:8000${job.result.annotated_video_url}`} download className="flex items-center gap-2 text-sm bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg transition-colors">
                  <Download size={16} /> Download Video
                </a>
              </div>
            </div>
          )}
          
          {job && job.status === 'failed' && (
            <div className="text-red-400 text-center">
              <AlertCircle size={48} className="mx-auto mb-2" />
              <p>Pipeline Failed</p>
              <p className="text-xs opacity-70 mt-2">{job.error}</p>
            </div>
          )}
        </div>
      </div>

      {/* 3. RIGHT COLUMN: Live Stats */}
      <div className="w-full xl:w-[320px] shrink-0 space-y-4">
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
      </div>
      
    </div>
  );
}
