"""
Rule-based генератор рекомендацій пасічнику на основі результату обробки.

Бере готовий job result (як його повертає VideoPipeline.get_result), застосовує
~12 правил із пасічницькими порогами та повертає список Recommendation.
"""
from dataclasses import asdict, dataclass


@dataclass
class Recommendation:
    severity: str          # 'info' | 'warning' | 'critical'
    icon: str              # emoji-ілюстрація для UI
    title: str             # коротке формулювання
    description: str       # пояснення для пасічника
    action: str | None     # дієва порада ("Перевірте температуру вулика")
    rule_id: str = ""      # ідентифікатор правила для UI прозорості
    details: dict | None = None  # числові дані що спровокували рекомендацію


# Усі пороги вкорінюються в `recomends.md` + експертну інтерпретацію PLOS ONE 2025.

# Defense
_DEF_WARNING_THRESHOLD = 1
_DEF_CRITICAL_THRESHOLD = 5

# Fanning як частка від загальної кількості детекцій поведінки
_FAN_WARN_RATIO = 0.30
_FAN_CRITICAL_RATIO = 0.50

# Traffic balance
_TRAFFIC_MIN_FOR_BALANCE_CHECK = 30   # абсолютний мінімум подій для аналізу балансу
_TRAFFIC_BALANCE_WARN_RATIO = 0.40
_NET_OUT_CRITICAL = 30                  # чистий вилет (OUT - IN) у штуках

# Foraging activity (events / minute)
_FOR_LOW_PER_MIN = 0.5
_FOR_HIGH_PER_MIN = 5.0

# Загальна активність трафіку (events / minute)
_TRAFFIC_LOW_PER_MIN = 2.0

# Pose-валідація (тільки для Approach B)
_POSE_LOW_RATIO = 0.50

# Washboarding як частка
_WASH_HIGH_RATIO = 0.20


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole > 0 else 0.0


def generate_recommendations(result: dict) -> list[Recommendation]:
    """
    Аналізує result-документ і повертає список рекомендацій пасічнику.
    Безпечний до відсутніх ключів (старі job-документи).
    """
    if not isinstance(result, dict):
        return []

    bs = result.get("behavior_summary") or {}
    foraging = int(bs.get("foraging_detections", 0))
    fanning = int(bs.get("fanning_detections", 0))
    washboarding = int(bs.get("washboarding_detections", 0))
    defense = int(bs.get("defense_detections", 0))
    total_behavior = foraging + fanning + washboarding + defense

    total_in = int(result.get("total_in", 0))
    total_out = int(result.get("total_out", 0))
    total_traffic = total_in + total_out

    duration_sec = float(result.get("duration_sec", 0.0))
    duration_min = max(duration_sec / 60.0, 1e-6)

    pose_ok = int(result.get("pose_confirmed_events", 0))
    pose_fb = int(result.get("fallback_events", 0))
    pose_rejected = int(result.get("pose_rejected_events", 0))
    approach = result.get("approach_used", "A")

    ramp_detected = bool(result.get("ramp_detected", True))

    recs: list[Recommendation] = []

    # 1–2. Defense
    if defense >= _DEF_CRITICAL_THRESHOLD:
        recs.append(Recommendation(
            severity="critical",
            icon="🚨",
            title=f"Часті захисні реакції ({defense})",
            description=(
                f"Зафіксовано {defense} епізодів захисної поведінки. Це сильний сигнал, "
                "що колонія перебуває під атакою (оси, бджоли-крадії) або в стресі."
            ),
            action="Огляньте льоток, зменшіть отвір, перевірте на наявність ос поблизу.",
            rule_id="defense_critical",
            details={"defense_count": defense, "threshold": _DEF_CRITICAL_THRESHOLD},
        ))
    elif defense >= _DEF_WARNING_THRESHOLD:
        recs.append(Recommendation(
            severity="critical",
            icon="🛡️",
            title="Виявлено захисну поведінку",
            description=(
                f"Бджоли сформували захисний кластер ({defense} епізод(и)). "
                "Це може бути спроба нападу або стресова реакція."
            ),
            action="Поспостерігайте за льотком найближчим часом, виявіть джерело загрози.",
            rule_id="defense_warning",
            details={"defense_count": defense, "threshold": _DEF_WARNING_THRESHOLD},
        ))

    # 3–4. Fanning intensity
    fan_ratio = _ratio(fanning, total_behavior)
    if fan_ratio >= _FAN_CRITICAL_RATIO:
        recs.append(Recommendation(
            severity="critical",
            icon="🔥",
            title="Можливий перегрів вулика",
            description=(
                f"{fan_ratio*100:.0f}% детекцій поведінки — вентиляція. Колонія агресивно "
                "охолоджує гніздо, що зазвичай означає високу температуру всередині."
            ),
            action="Перевірте температуру вулика, забезпечте тінь і доступ до води.",
            rule_id="fanning_critical",
            details={"fanning_ratio": round(fan_ratio, 3), "fanning_count": fanning, "total_behavior": total_behavior, "threshold": _FAN_CRITICAL_RATIO},
        ))
    elif fan_ratio >= _FAN_WARN_RATIO:
        recs.append(Recommendation(
            severity="warning",
            icon="🌬️",
            title="Інтенсивна вентиляція",
            description=(
                f"{fan_ratio*100:.0f}% активності — вентиляція. Колонія активно регулює "
                "температуру/вологість, а також може поширювати феромони."
            ),
            action="Якщо триває довго — перевірте температуру і вологість всередині вулика.",
            rule_id="fanning_warning",
            details={"fanning_ratio": round(fan_ratio, 3), "fanning_count": fanning, "total_behavior": total_behavior, "threshold": _FAN_WARN_RATIO},
        ))

    # 5. Дисбаланс трафіку (можливе роєння або проблема з маткою)
    if total_traffic >= _TRAFFIC_MIN_FOR_BALANCE_CHECK:
        max_dir = max(total_in, total_out)
        balance = abs(total_in - total_out) / max_dir if max_dir > 0 else 0.0
        if balance >= _TRAFFIC_BALANCE_WARN_RATIO:
            direction = "більше вилетіло" if total_out > total_in else "більше повернулось"
            recs.append(Recommendation(
                severity="warning",
                icon="⚖️",
                title="Дисбаланс трафіку",
                description=(
                    f"IN: {total_in}, OUT: {total_out} ({balance*100:.0f}% {direction}). "
                    "Стійкий дисбаланс може вказувати на роєння, проблеми з маткою або "
                    "на видобуток ресурсів далеко від вулика."
                ),
                action="Огляньте розплід і королеву, спостережіть за льотком наступні кілька днів.",
                rule_id="traffic_imbalance",
                details={"total_in": total_in, "total_out": total_out, "balance": round(balance, 3), "threshold": _TRAFFIC_BALANCE_WARN_RATIO},
            ))

    # 6. Масовий вилет
    net_out = total_out - total_in
    if net_out >= _NET_OUT_CRITICAL:
        recs.append(Recommendation(
            severity="critical",
            icon="🐝",
            title=f"Масовий вилет без повернення (×{net_out})",
            description=(
                f"Бджоли вилетіли на {net_out} більше, ніж повернулися. Це класичний "
                "індикатор роєння або раптової міграції колонії."
            ),
            action="Терміново перевірте вулик — можливе роєння або зникнення матки.",
            rule_id="mass_departure",
            details={"net_out": net_out, "total_in": total_in, "total_out": total_out, "threshold": _NET_OUT_CRITICAL},
        ))

    # 7–8. Foraging activity
    foraging_per_min = foraging / duration_min
    if foraging_per_min >= _FOR_HIGH_PER_MIN:
        recs.append(Recommendation(
            severity="info",
            icon="🌻",
            title="Висока активність фуражування",
            description=(
                f"~{foraging_per_min:.1f} фуражних детекцій за хвилину — кормова база "
                "достатня, погода сприятлива, колонія ефективно збирає нектар і пилок."
            ),
            action=None,
            rule_id="foraging_high",
            details={"foraging_per_min": round(foraging_per_min, 2), "threshold": _FOR_HIGH_PER_MIN},
        ))
    elif foraging_per_min < _FOR_LOW_PER_MIN and duration_min > 1:
        recs.append(Recommendation(
            severity="info",
            icon="🥀",
            title="Знижена активність фуражування",
            description=(
                f"Лише ~{foraging_per_min:.2f} фуражних детекцій/хв. Можливі причини: "
                "негода, дефіцит цвітіння поблизу або період між хабарами."
            ),
            action="За тривалого зниження — розгляньте підгодівлю або перенесення вулика.",
            rule_id="foraging_low",
            details={"foraging_per_min": round(foraging_per_min, 2), "threshold": _FOR_LOW_PER_MIN},
        ))

    # 9. Загальна активність
    traffic_per_min = total_traffic / duration_min
    if traffic_per_min < _TRAFFIC_LOW_PER_MIN and duration_min > 1:
        recs.append(Recommendation(
            severity="info",
            icon="💤",
            title="Низький загальний трафік",
            description=(
                f"~{traffic_per_min:.1f} перетинів льотка/хв. Якщо це не передранкові години "
                "та не плохий день — можливо колонія слабка або у стані стресу."
            ),
            action="Зважте вулик і перевірте загальний стан розплоду.",
            rule_id="traffic_low",
            details={"traffic_per_min": round(traffic_per_min, 2), "threshold": _TRAFFIC_LOW_PER_MIN},
        ))

    # 10. Pose-валідація (тільки для Approach B)
    if approach == "B":
        pose_total = pose_ok + pose_fb
        pose_ratio = _ratio(pose_ok, pose_total)
        if pose_total > 5 and pose_ratio < _POSE_LOW_RATIO:
            recs.append(Recommendation(
                severity="info",
                icon="📐",
                title="Низька якість pose-валідації",
                description=(
                    f"Лише {pose_ratio*100:.0f}% подій підтверджено вектором пози. "
                    "Це впливає на точність методу Б — можлива слабка видимість бджіл "
                    "або погана позиція камери."
                ),
                action="Перевірте кут камери, освітлення і фокус.",
                rule_id="pose_quality_low",
                details={"pose_ratio": round(pose_ratio, 3), "pose_ok": pose_ok, "pose_total": pose_total, "threshold": _POSE_LOW_RATIO},
            ))

        # 10b. Ефективність pose-фільтра — скільки фейкових перетинів відсіяно
        a_total = pose_ok + pose_fb + pose_rejected  # скільки зарахував би Approach A
        if a_total > 5:
            rejected_ratio = _ratio(pose_rejected, a_total)
            if rejected_ratio >= 0.20:
                recs.append(Recommendation(
                    severity="info",
                    icon="🎯",
                    title=f"Pose-фільтр відсіяв {rejected_ratio*100:.0f}% перетинів",
                    description=(
                        f"Approach A нарахував би {a_total} подій, Approach B відсіяв "
                        f"{pose_rejected} як хибні (вектор пози не співпав з рухом). "
                        "Це показник, наскільки точніший метод Б на цьому відео."
                    ),
                    action=None,
                    rule_id="pose_filter_efficiency",
                    details={"rejected_ratio": round(rejected_ratio, 3), "rejected": pose_rejected, "a_total": a_total},
                ))

    # 11. Washboarding
    wash_ratio = _ratio(washboarding, total_behavior)
    if wash_ratio >= _WASH_HIGH_RATIO:
        recs.append(Recommendation(
            severity="info",
            icon="🧹",
            title="Активне полірування льотка",
            description=(
                f"{wash_ratio*100:.0f}% активності — ритмічні рухи на льотку. "
                "Це поведінка догляду за чистотою; не вимагає втручання."
            ),
            action=None,
            rule_id="washboarding_high",
            details={"wash_ratio": round(wash_ratio, 3), "washboarding_count": washboarding, "total_behavior": total_behavior, "threshold": _WASH_HIGH_RATIO},
        ))

    # 12. Льоток не виявлено
    if not ramp_detected:
        recs.append(Recommendation(
            severity="warning",
            icon="📷",
            title="Льоток не виявлено",
            description=(
                "Модель не змогла знайти прилітну дошку у відео. Показники підрахунку "
                "та поведінки можуть бути ненадійними."
            ),
            action="Перевірте позицію камери: льоток має бути добре видно у кадрі.",
            rule_id="ramp_not_detected",
            details=None,
        ))

    return recs


def recommendations_to_dicts(recs: list[Recommendation]) -> list[dict]:
    """Серіалізація для збереження в Mongo / повернення через API."""
    return [asdict(r) for r in recs]
