import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { UploadCloud, Video, Settings, ChevronRight, ChevronDown, Sliders, Play, Cpu } from 'lucide-react';
import type { ProcessConfig, VizConfig } from '../types';

const VIZ_LABELS: Record<string, string> = {
  show_boxes:         'Рамки',
  show_ids:           'ID треків',
  show_confidence:    'Впевненість',
  show_keypoints:     'Ключові точки',
  show_ramp:          'Рампа',
  show_behaviors:     'Поведінки',
  show_counting_line: 'Лінія підрахунку',
  show_stats_overlay: 'Статистика',
  show_tracks:        'Траєкторії',
  show_orientation:   'Орієнтація',
  show_recent_events: 'Останні події',
};

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

function Slider({
  label, value, min, max, step, onChange, unit = '',
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; unit?: string;
}) {
  return (
    <div>
      <label className="text-xs text-gray-400 flex justify-between mb-1">
        <span>{label}</span>
        <span className="text-gray-200">{value}{unit}</span>
      </label>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full accent-[var(--accent)] h-1"
      />
    </div>
  );
}

export default function JobConfigPanel({
  file, setFile, config, setConfig, vizConfig, setVizConfig,
  testVideos, isProcessing, jobId, onStart, onStartTest,
}: JobConfigPanelProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showBehavior, setShowBehavior] = useState(false);
  const [showViz, setShowViz] = useState(false);

  type ModelInfo = {
    name: string;
    arch?: string | null;
    variant?: string | null;
    task?: string | null;
    imgsz?: number | null;
    trained_with_half?: boolean;
  };

  const { data: availableModels = [] } = useQuery<ModelInfo[]>({
    queryKey: ['models'],
    queryFn: async () => {
      const resp = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/models`);
      return resp.json();
    },
  });

  const c = config as any;
  const set = (patch: Partial<typeof c>) => setConfig({ ...config, ...patch } as ProcessConfig);

  const selectedModel = availableModels.find(m => m.name === c.model_name)
    ?? availableModels.find(m => m.name === 'bee_pose')
    ?? availableModels[0];

  const canStart = (!!file || !!jobId) && !isProcessing;

  return (
    <div className="card space-y-4">
      <h2 className="text-base font-semibold border-b border-gray-700 pb-2 flex items-center gap-2">
        <UploadCloud size={18} className="text-[var(--accent)]" /> Завантаження відео
      </h2>

      {/* Зона завантаження файлу */}
      <div className="border-2 border-dashed border-gray-700 hover:border-[var(--accent)] cursor-pointer rounded-xl p-5 text-center transition-colors bg-[var(--bg-panel)] relative">
        <input type="file" accept="video/*" onChange={e => setFile(e.target.files?.[0] || null)} className="absolute inset-0 opacity-0 cursor-pointer" />
        {!file ? (
          <div className="text-gray-400 text-sm">
            <Video size={32} className="mx-auto mb-2 opacity-40" />
            <p>Перетягніть або натисніть</p>
            <p className="text-xs mt-1 opacity-60">MP4, AVI, MOV · до 500 МБ</p>
          </div>
        ) : (
          <div className="text-green-400 text-sm font-medium break-all">{file.name} ({(file.size / 1024 / 1024).toFixed(1)} МБ)</div>
        )}
      </div>

      {/* Тестові відео */}
      <div className="flex flex-wrap gap-2">
        {testVideos.map(tv => (
          <button
            key={tv}
            onClick={() => onStartTest(tv)}
            disabled={isProcessing}
            className="bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs py-1.5 px-3 rounded-lg flex-1 min-w-[110px] transition-colors disabled:opacity-50 flex items-center justify-center gap-1.5"
          >
            <Play size={11} /> {tv}
          </button>
        ))}
        {testVideos.length === 0 && <span className="text-xs text-gray-600">Тестових відео немає</span>}
      </div>

      {/* Вибір моделі */}
      {availableModels.length > 1 && (
        <div>
          <label className="text-xs text-gray-400 mb-1 flex items-center gap-1.5">
            <Cpu size={12} /> Модель детекції
          </label>
          <select
            value={c.model_name || ''}
            onChange={e => set({ model_name: e.target.value || null })}
            className="w-full bg-[var(--bg-panel)] border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 focus:border-[var(--accent)] focus:outline-none"
          >
            <option value="">За замовч. (bee_pose)</option>
            {availableModels.map(m => (
              <option key={m.name} value={m.name}>
                {m.name}
                {m.variant ? ` · yolo11${m.variant}` : ''}
                {m.imgsz ? ` · ${m.imgsz}пкс` : ''}
              </option>
            ))}
          </select>
          {selectedModel && (
            <p className="text-[11px] text-gray-500 mt-1">
              {selectedModel.task ?? '—'} ·{' '}
              навчено на {selectedModel.imgsz ?? '?'}пкс ·{' '}
              {selectedModel.trained_with_half ? 'FP16' : 'FP32'}
            </p>
          )}
        </div>
      )}

      {/* Трекер */}
      <div>
        <label className="text-xs text-gray-400 mb-1 block">Трекер</label>
        <div className="flex gap-4">
          {['bytetrack', 'ocsort'].map(t => (
            <label key={t} className="flex items-center gap-2 text-sm cursor-pointer">
              <input type="radio" checked={config.tracker_name === t} onChange={() => set({ tracker_name: t })} className="accent-[var(--accent)]" />
              {t === 'bytetrack' ? 'ByteTrack' : 'OC-SORT'}
            </label>
          ))}
        </div>
      </div>

      {/* Метод підрахунку */}
      <div>
        <label className="text-xs text-gray-400 mb-1 block">Метод підрахунку</label>
        <div className="flex flex-col gap-2">
          <label className="flex items-start gap-2 text-sm cursor-pointer">
            <input type="radio" checked={config.approach === 'A'} onChange={() => set({ approach: 'A' })} className="accent-[var(--accent)] mt-0.5" />
            <span>
              <span className="font-medium">А</span> — За траєкторією
              <span className="block text-[11px] text-gray-500 mt-0.5">Фіксує перетин лінії за рухом центра бджоли</span>
            </span>
          </label>
          <label className="flex items-start gap-2 text-sm cursor-pointer">
            <input type="radio" checked={config.approach === 'B'} onChange={() => set({ approach: 'B' })} className="accent-[var(--accent)] mt-0.5" />
            <span>
              <span className="font-medium text-[var(--accent)]">Б</span> — З валідацією пози
              <span className="block text-[11px] text-gray-500 mt-0.5">Перевіряє напрямок вектора голова→жало</span>
            </span>
          </label>
        </div>
      </div>

      {/* Детекція і трекінг */}
      <button
        className="flex items-center justify-between w-full text-xs font-semibold text-gray-400 py-2 border-t border-gray-700"
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        <span className="flex items-center gap-2"><Settings size={14} /> Детекція і трекінг</span>
        {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      {showAdvanced && (
        <div className="space-y-3 bg-[var(--bg-panel)] p-3 rounded-lg border border-gray-800">
          <Slider label="Поріг впевненості детекції" value={config.conf_threshold ?? 0.20} min={0.1} max={0.9} step={0.05} onChange={v => set({ conf_threshold: v })} />
          <Slider label="Поріг впевненості ключових точок" value={config.kp_conf_threshold ?? 0.5} min={0.1} max={0.9} step={0.05} onChange={v => set({ kp_conf_threshold: v })} />
          <Slider label="Допустимий кут відхилення" value={config.angle_threshold_deg ?? 60} min={15} max={90} step={5} onChange={v => set({ angle_threshold_deg: v })} unit="°" />
          <Slider label="Довжина хвоста треку" value={config.track_tail_length ?? 30} min={10} max={100} step={5} onChange={v => set({ track_tail_length: v })} unit=" кадр" />
          <Slider label="Інтервал виявлення рампи" value={config.ramp_detect_interval ?? 30} min={5} max={60} step={5} onChange={v => set({ ramp_detect_interval: v })} unit=" кадр" />

          {/* Інференс */}
          <div className="pt-2 border-t border-gray-800">
            <p className="text-[11px] text-gray-500 mb-2">Інференс (VRAM і швидкість)</p>
            <div>
              <label className="text-xs text-gray-400 flex justify-between mb-1">
                <span>Розмір зображення</span>
                <span className="text-gray-200">
                  {c.imgsz ?? `авто · ${selectedModel?.imgsz ?? '?'}пкс`}
                </span>
              </label>
              <select
                value={c.imgsz ?? ''}
                onChange={e => set({ imgsz: e.target.value ? Number(e.target.value) : null })}
                className="w-full bg-[var(--bg-page)] border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 focus:border-[var(--accent)] focus:outline-none"
              >
                <option value="">Авто (за моделлю)</option>
                <option value="640">640 — найшвидше, нижча точність</option>
                <option value="960">960</option>
                <option value="1280">1280 — баланс</option>
                <option value="1600">1600</option>
                <option value="1920">1920 — найкраще для бджіл</option>
              </select>
            </div>

            <div className="mt-2">
              <label className="text-xs text-gray-400 flex justify-between mb-1">
                <span>Розмір батчу</span>
                <span className="text-gray-200">
                  {c.batch_size ?? (c.half_precision ? 'авто · 4' : 'авто · 2')}
                </span>
              </label>
              <input
                type="range"
                min={1} max={32} step={1}
                value={c.batch_size ?? (c.half_precision ? 4 : 2)}
                onChange={e => set({ batch_size: Number(e.target.value) })}
                className="w-full accent-[var(--accent)] h-1"
              />
              <p className="text-[10px] text-gray-600 mt-1">
                MX550 (4 ГБ): 2 при 1920пкс FP32. T4 (16 ГБ): 8–16.
              </p>
            </div>

            <label className="flex items-center gap-2 text-xs text-gray-400 mt-2 cursor-pointer">
              <input
                type="checkbox"
                checked={!!c.half_precision}
                onChange={e => set({ half_precision: e.target.checked })}
                className="accent-[var(--accent)]"
              />
              FP16 (половинна точність) — швидше, але може давати NaN на деяких вагах
            </label>
          </div>
        </div>
      )}

      {/* Пороги поведінки */}
      <button
        className="flex items-center justify-between w-full text-xs font-semibold text-gray-400 py-2 border-t border-gray-700"
        onClick={() => setShowBehavior(!showBehavior)}
      >
        <span className="flex items-center gap-2"><Settings size={14} /> Пороги поведінки</span>
        {showBehavior ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      {showBehavior && (
        <div className="space-y-3 bg-[var(--bg-panel)] p-3 rounded-lg border border-gray-800">
          <p className="text-xs text-gray-500">Пороги для евристичного класифікатора (пкс/с).</p>
          <Slider label="Мін. швидкість фуражування" value={Number(c.behavior_foraging_speed_min ?? 100)} min={30} max={300} step={10} onChange={v => set({ behavior_foraging_speed_min: v })} unit=" пкс/с" />
          <Slider label="Макс. швидкість вентиляції" value={Number(c.behavior_fanning_speed_max ?? 15)} min={5} max={50} step={5} onChange={v => set({ behavior_fanning_speed_max: v })} unit=" пкс/с" />
          <Slider label="Мін. тривалість вентиляції" value={Number(c.behavior_fanning_duration_min ?? 2)} min={0.5} max={10} step={0.5} onChange={v => set({ behavior_fanning_duration_min: v })} unit=" с" />
          <Slider label="Мін. швидкість охорони" value={Number(c.behavior_guarding_speed_min ?? 15)} min={5} max={80} step={5} onChange={v => set({ behavior_guarding_speed_min: v })} unit=" пкс/с" />
          <Slider label="Макс. швидкість охорони" value={Number(c.behavior_guarding_speed_max ?? 80)} min={30} max={200} step={10} onChange={v => set({ behavior_guarding_speed_max: v })} unit=" пкс/с" />
        </div>
      )}

      {/* Візуалізація */}
      <button
        className="flex items-center justify-between w-full text-xs font-semibold text-gray-400 py-2 border-t border-gray-700"
        onClick={() => setShowViz(!showViz)}
      >
        <span className="flex items-center gap-2"><Sliders size={14} /> Візуалізація</span>
        {showViz ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      {showViz && (
        <div className="grid grid-cols-2 gap-2 bg-[var(--bg-panel)] p-3 rounded-lg border border-gray-800 text-xs">
          {Object.keys(vizConfig).map(k => (
            <label key={k} className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={(vizConfig as any)[k]}
                onChange={e => setVizConfig({ ...vizConfig, [k]: e.target.checked })}
                className="accent-[var(--accent)] rounded"
              />
              <span className="truncate" title={k}>{VIZ_LABELS[k] ?? k.replace('show_', '')}</span>
            </label>
          ))}
        </div>
      )}

      <button
        disabled={!canStart}
        onClick={onStart}
        title={!file && !jobId ? 'Оберіть або завантажте відео' : undefined}
        className={`btn-primary w-full flex justify-center items-center gap-2 mt-2 ${!canStart ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        {isProcessing
          ? <span className="animate-pulse">Обробка…</span>
          : <><Play size={16} /> Запустити аналіз</>
        }
      </button>
      {!file && !jobId && !isProcessing && (
        <p className="text-[11px] text-gray-600 text-center -mt-2">Оберіть файл або тестове відео</p>
      )}
    </div>
  );
}
