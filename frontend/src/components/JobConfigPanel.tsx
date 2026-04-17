import { useState } from 'react';
import { UploadCloud, Video, Settings, ChevronRight, ChevronDown, Sliders, Play } from 'lucide-react';
import type { ProcessConfig, VizConfig } from '../types';

interface JobConfigPanelProps {
  file: File | null;
  setFile: (file: File | null) => void;
  config: ProcessConfig;
  setConfig: (config: ProcessConfig) => void;
  vizConfig: VizConfig;
  setVizConfig: (config: VizConfig) => void;
  testVideos: string[];
  isProcessing: boolean;
  jobId: string | null;
  onStart: () => void;
  onStartTest: (filename: string) => void;
}

export default function JobConfigPanel({
  file, setFile, config, setConfig, vizConfig, setVizConfig, 
  testVideos, isProcessing, jobId, onStart, onStartTest
}: JobConfigPanelProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showViz, setShowViz] = useState(false);

  return (
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
             onClick={() => onStartTest(tv)} 
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
          Line Position <span>{config.line_position?.toFixed(2)}</span>
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
        onClick={onStart}
        className={`btn-primary w-full flex justify-center items-center gap-2 mt-4 ${(!file && !jobId) || isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}>
        {isProcessing ? <span className="animate-pulse">Processing...</span> : <><Play size={18} /> Start Pipeline with File</>}
      </button>
    </div>
  );
}
