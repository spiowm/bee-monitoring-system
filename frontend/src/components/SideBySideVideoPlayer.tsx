import { useEffect, useRef, useState } from 'react';
import { Play, Pause } from 'lucide-react';

export interface VideoSource {
  url: string;
  label: string;
  sublabel?: string;
  borderClass?: string;
  textClass?: string;
}

interface Props {
  videos: VideoSource[];
}

const SPEEDS = [0.5, 1, 1.5, 2, 4] as const;
const apiBase = (import.meta.env.VITE_API_URL as string | undefined) || window.location.origin;
const resolve = (url: string) => (url.startsWith('http') ? url : `${apiBase}${url}`);

export default function SideBySideVideoPlayer({ videos }: Props) {
  const refs = useRef<(HTMLVideoElement | null)[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState<number>(1);
  const syncing = useRef(false);

  useEffect(() => {
    const master = refs.current[0];
    if (!master) return;

    const handleTimeUpdate = () => {
      if (syncing.current) return;
      syncing.current = true;
      for (let i = 1; i < refs.current.length; i++) {
        const slave = refs.current[i];
        if (!slave) continue;
        if (Math.abs(slave.currentTime - master.currentTime) > 0.15) {
          slave.currentTime = master.currentTime;
        }
      }
      requestAnimationFrame(() => {
        syncing.current = false;
      });
    };

    master.addEventListener('timeupdate', handleTimeUpdate);
    master.addEventListener('seeking', handleTimeUpdate);
    return () => {
      master.removeEventListener('timeupdate', handleTimeUpdate);
      master.removeEventListener('seeking', handleTimeUpdate);
    };
  }, [videos.length]);

  const togglePlay = () => {
    const master = refs.current[0];
    if (!master) return;
    if (master.paused) {
      refs.current.forEach(v => v?.play());
      setPlaying(true);
    } else {
      refs.current.forEach(v => v?.pause());
      setPlaying(false);
    }
  };

  const applyRate = (r: number) => {
    setRate(r);
    refs.current.forEach(v => {
      if (v) v.playbackRate = r;
    });
  };

  const stepFrame = (frames: number) => {
    const master = refs.current[0];
    if (!master) return;
    master.pause();
    master.currentTime = Math.max(0, Math.min(master.duration || 0, master.currentTime + frames * (1 / 30)));
    setPlaying(false);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const master = refs.current[0];
    if (!master) return;

    const target = e.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') return;

    switch (e.key) {
      case ' ':
      case 'k':
      case 'K':
        e.preventDefault();
        togglePlay();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        master.currentTime = Math.max(0, master.currentTime - 5);
        break;
      case 'ArrowRight':
        e.preventDefault();
        master.currentTime = Math.min(master.duration || 0, master.currentTime + 5);
        break;
      case 'j':
      case 'J':
        e.preventDefault();
        master.currentTime = Math.max(0, master.currentTime - 10);
        break;
      case 'l':
      case 'L':
        e.preventDefault();
        master.currentTime = Math.min(master.duration || 0, master.currentTime + 10);
        break;
      case ',':
        e.preventDefault();
        stepFrame(-1);
        break;
      case '.':
        e.preventDefault();
        stepFrame(1);
        break;
      default:
        if (/^[0-9]$/.test(e.key)) {
          e.preventDefault();
          const pct = parseInt(e.key, 10) / 10;
          master.currentTime = (master.duration || 0) * pct;
        }
    }
  };

  const cols = videos.length === 1 ? 'md:grid-cols-1' : videos.length === 2 ? 'md:grid-cols-2' : 'md:grid-cols-3';

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      onKeyDown={onKeyDown}
      onClick={() => containerRef.current?.focus()}
      className="space-y-3 outline-none"
    >
      <div className={`grid grid-cols-1 ${cols} gap-3`}>
        {videos.map((v, i) => (
          <div
            key={i}
            className={`bg-[var(--bg-panel)] rounded-xl border overflow-hidden ${v.borderClass ?? 'border-gray-700'}`}
          >
            <div className="px-3 py-2 text-xs font-semibold uppercase tracking-wide border-b border-gray-800 flex items-center justify-between">
              <span className={v.textClass ?? 'text-gray-300'}>{v.label}</span>
              {v.sublabel && (
                <span className="text-gray-500 normal-case font-normal">{v.sublabel}</span>
              )}
            </div>
            <video
              ref={el => { refs.current[i] = el; }}
              src={resolve(v.url)}
              controls={i === 0}
              muted={i !== 0}
              preload="metadata"
              className="w-full bg-black aspect-video"
            />
          </div>
        ))}
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3">
        <button
          type="button"
          onClick={togglePlay}
          className="px-5 py-2 bg-[var(--accent)] hover:brightness-110 text-black font-semibold rounded-lg flex items-center gap-2 transition"
        >
          {playing ? <Pause size={16} /> : <Play size={16} />}
          {playing ? 'Pause всі' : 'Play всі (синхронно)'}
        </button>

        <div
          className="flex items-center gap-1 text-xs"
          title="Гарячі клавіші: Space — pause, ← → 5с, J/L 10с, , . — кадр, 0–9 — стрибок"
        >
          <span className="text-gray-500 uppercase tracking-wider mr-2 ml-4">Кадр</span>
          <button
            type="button"
            onClick={() => stepFrame(-1)}
            title="Попередній кадр (,)"
            className="px-2 py-1 rounded font-mono text-gray-300 bg-gray-800 hover:bg-gray-700 transition"
          >
            -1
          </button>
          <button
            type="button"
            onClick={() => stepFrame(1)}
            title="Наступний кадр (.)"
            className="px-2 py-1 rounded font-mono text-gray-300 bg-gray-800 hover:bg-gray-700 transition"
          >
            +1
          </button>

          <span className="text-gray-500 uppercase tracking-wider mx-2">Швидкість</span>
          {SPEEDS.map(s => (
            <button
              key={s}
              type="button"
              onClick={() => applyRate(s)}
              className={`px-2.5 py-1 rounded font-mono transition ${
                rate === s
                  ? 'bg-[var(--accent)] text-black font-bold'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
