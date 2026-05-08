import { Info } from 'lucide-react';

interface BehaviorInfo {
  key: string;
  label: string;
  short: string;
  description: string;
  thresholds: string;
  cssVar: string;
}

const BEHAVIORS: BehaviorInfo[] = [
  {
    key: 'foraging',
    label: 'Фуражування',
    short: 'FOR',
    description: 'Бджола повертається з пошуку нектару або вилітає за ним. Швидкий, цілеспрямований рух.',
    thresholds: 'Швидкість > 100 пкс/с',
    cssVar: '--behavior-foraging',
  },
  {
    key: 'fanning',
    label: 'Вентиляція',
    short: 'FAN',
    description: 'Майже нерухома бджола з тілом, спрямованим до льотка, що активно махає крилами. Регулює температуру/вологість вулика та поширює феромони.',
    thresholds: 'Зміщення < 10 пкс, тривалість > 1 с, кут до льотка ±90°',
    cssVar: '--behavior-fanning',
  },
  {
    key: 'washboarding',
    label: 'Полірування',
    short: 'POL',
    description: 'Ритмічні коливальні рухи вперед-назад на льотку — бджоли очищують і полірують поверхню деревини після фуражування.',
    thresholds: 'Швидкість 0–60 пкс/с, тривалість > 2 с, частота зміни напряму > 2 Гц',
    cssVar: '--behavior-washboarding',
  },
  {
    key: 'defense',
    label: 'Захист',
    short: 'DEF',
    description: 'Кластер охоронців, які вишиковуються довкола чужинця (оси, бджоли-крадія) та націлюють тіла на нього. Сигналізує про загрозу для колонії.',
    thresholds: '≥2 захисниці в радіусі 2 довжин бджоли, кут на ціль ±45°, тривалість >1 с',
    cssVar: '--behavior-defense',
  },
];

interface Props {
  compact?: boolean;
}

export default function BehaviorLegend({ compact = false }: Props) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Поведінкові класи</span>
        <Info size={12} className="text-gray-600" />
      </div>
      <div className={compact ? 'flex flex-wrap gap-1.5' : 'grid grid-cols-2 gap-1.5'}>
        {BEHAVIORS.map(b => (
          <details
            key={b.key}
            className="group relative bg-[var(--bg-panel)] rounded-md border border-gray-800 hover:border-gray-700 transition-colors"
          >
            <summary
              className="flex items-center gap-2 px-2 py-1.5 cursor-pointer list-none"
              title={b.description}
            >
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ background: `var(${b.cssVar})` }}
              />
              <span className="text-xs text-gray-300 truncate">{b.label}</span>
            </summary>
            <div className="absolute z-10 left-0 right-0 mt-1 p-3 bg-[var(--bg-card)] border border-gray-700 rounded-lg shadow-xl text-xs space-y-2 min-w-[240px]">
              <div className="flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ background: `var(${b.cssVar})` }}
                />
                <span className="font-semibold text-gray-100">{b.label}</span>
                <span className="text-[10px] text-gray-500 font-mono">{b.short}</span>
              </div>
              <p className="text-gray-300 leading-relaxed">{b.description}</p>
              <p className="text-[10px] text-gray-500 italic border-t border-gray-800 pt-2">
                Критерій: {b.thresholds}
              </p>
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}
