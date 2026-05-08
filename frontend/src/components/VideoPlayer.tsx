import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';

const SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4] as const;

export interface VideoPlayerHandle {
  el: HTMLVideoElement | null;
  setRate: (rate: number) => void;
}

interface Props extends Omit<React.VideoHTMLAttributes<HTMLVideoElement>, 'onRateChange'> {
  src: string;
  /** Якщо false — приховати speed controls (наприклад для slave-відео в SideBySidePlayer). */
  showSpeedControls?: boolean;
  /** Зовнішній колбек, коли ставка змінюється (для синхронізації між декількома плеєрами). */
  onPlaybackRateChange?: (rate: number) => void;
  /** Початкова швидкість. */
  initialRate?: number;
  /** Прибирає браузерні controls + keyboard shortcuts (для slave у SideBySide). */
  passive?: boolean;
}

const VideoPlayer = forwardRef<VideoPlayerHandle, Props>(function VideoPlayer(
  {
    src,
    showSpeedControls = true,
    onPlaybackRateChange,
    initialRate = 1,
    passive = false,
    className,
    controls,
    muted,
    preload,
    ...rest
  },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [rate, setRateState] = useState<number>(initialRate);

  useImperativeHandle(ref, () => ({
    get el() {
      return videoRef.current;
    },
    setRate: (r: number) => applyRate(r, false),
  }));

  const applyRate = (r: number, notify: boolean = true) => {
    setRateState(r);
    if (videoRef.current) videoRef.current.playbackRate = r;
    if (notify) onPlaybackRateChange?.(r);
  };

  const stepFrame = (frames: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    v.currentTime = Math.max(0, Math.min(v.duration || 0, v.currentTime + frames * (1 / 30)));
  };

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = initialRate;
  }, [initialRate, src]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (passive) return;
    const v = videoRef.current;
    if (!v) return;

    const target = e.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') return;

    const speedIdx = SPEEDS.indexOf(rate as (typeof SPEEDS)[number]);

    switch (e.key) {
      case ' ':
      case 'k':
      case 'K':
        e.preventDefault();
        if (v.paused) v.play(); else v.pause();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        v.currentTime = Math.max(0, v.currentTime - 5);
        break;
      case 'ArrowRight':
        e.preventDefault();
        v.currentTime = Math.min(v.duration || 0, v.currentTime + 5);
        break;
      case 'j':
      case 'J':
        e.preventDefault();
        v.currentTime = Math.max(0, v.currentTime - 10);
        break;
      case 'l':
      case 'L':
        e.preventDefault();
        v.currentTime = Math.min(v.duration || 0, v.currentTime + 10);
        break;
      case ',':
        e.preventDefault();
        stepFrame(-1);
        break;
      case '.':
        e.preventDefault();
        stepFrame(1);
        break;
      case 'm':
      case 'M':
        e.preventDefault();
        v.muted = !v.muted;
        break;
      case '+':
      case '=':
        e.preventDefault();
        if (speedIdx >= 0 && speedIdx < SPEEDS.length - 1) applyRate(SPEEDS[speedIdx + 1]);
        break;
      case '-':
      case '_':
        e.preventDefault();
        if (speedIdx > 0) applyRate(SPEEDS[speedIdx - 1]);
        break;
      default:
        if (/^[0-9]$/.test(e.key)) {
          e.preventDefault();
          const pct = parseInt(e.key, 10) / 10;
          v.currentTime = (v.duration || 0) * pct;
        }
    }
  };

  return (
    <div
      ref={containerRef}
      tabIndex={passive ? -1 : 0}
      onKeyDown={onKeyDown}
      className={`flex flex-col gap-2 outline-none ${className ?? ''}`}
    >
      <video
        ref={videoRef}
        src={src}
        controls={passive ? false : (controls ?? true)}
        muted={passive ? true : muted}
        preload={preload ?? 'metadata'}
        className="w-full bg-black rounded-lg"
        onClick={() => containerRef.current?.focus()}
        {...rest}
      />
      {showSpeedControls && !passive && (
        <div
          className="flex items-center justify-between gap-2 text-xs"
          title="Гарячі клавіші: Space — pause, ← → 5с, J/L 10с, , . — кадр, M — mute, 0–9 — стрибок, +/- — швидкість"
        >
          <div className="flex items-center gap-4 w-full justify-between">
            <div className="flex gap-1 items-center">
               <span className="text-gray-500 uppercase tracking-wider mr-1" title="Покадрово (, / .)">Кадр</span>
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
            </div>
            
            <div className="flex gap-1 items-center">
              <span className="text-gray-500 uppercase tracking-wider mr-1">Швидкість</span>
              <div className="flex gap-1">
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
        </div>
      )}
    </div>
  );
});

export default VideoPlayer;
