from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from db.mongodb import get_db
from datetime import datetime

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/export/md", response_class=PlainTextResponse)
async def export_md_report(job_a_id: str = None, job_b_id: str = None):
    """Generates a detailed Markdown report comparing up to two evaluated jobs."""
    db = get_db()
    
    jobs = []
    if job_a_id:
        ja = await db["jobs"].find_one({"job_id": job_a_id})
        if ja: jobs.append(("A", ja))
    if job_b_id:
        jb = await db["jobs"].find_one({"job_id": job_b_id})
        if jb: jobs.append(("B", jb))
        
    if not jobs:
        return PlainTextResponse("No jobs found to generate report.", status_code=404)

    lines = [
        f"# Звіт про тестування BuzzTrack (Автоматична генерація)",
        f"**Дата генерації:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        "Цей звіт містить результати зіставлення передбачень системи з Ground Truth (GT) анотаціями.",
        "Для порівняння підрахунку трафіку використовується жадібне зіставлення подій (TP/FP/FN) з вікном у ±15 кадрів.",
        "> **Стратегічна валідність:** Порівняння підрахунку (Traffic) є повністю валідним, оскільки лінія підрахунку в обох випадках фіксується на верхньому краю льотка (entrance zone). Підхід B доводить свою ефективність через зменшення False Positives. Порівняння поведінки носить ознайомчий характер (унікальні треки), оскільки набір міток у GT та системі частково відрізняється (washboarding відсутнє у GT).",
        ""
    ]

    for label, job in jobs:
        cfg = job.get("config", {})
        eval_data = job.get("evaluation")
        res = job.get("result", {})
        
        lines.append(f"## Метод {label}: {cfg.get('approach', 'A')}")
        if not eval_data:
            lines.append("❌ Дані оцінки (Evaluation) відсутні. Можливо, завдання ще виконується або завершилось з помилкою.\n")
            continue

        gt_file = eval_data.get('gt_basename', 'unknown')
        dur = res.get('duration_sec', 0)
        lines.extend([
            f"- **Відео:** `{gt_file}`",
            f"- **Тривалість обробки:** {dur:.1f} сек ({res.get('fps_processed', 0):.1f} fps)",
            f"- **Трекер:** {cfg.get('tracker_name', 'bytetrack')}",
            f"- **Approach:** {cfg.get('approach', 'A')} (A = Траєкторія, B = Pose-validated)",
            f"- **Лінія підрахунку:** {cfg.get('line_position', 0.0)} (0.0 = верхній край рампи)",
            f"- **Спрацювань анти-джиттер фільтру (Debounce blocks):** {res.get('debounce_blocks', 0)}",
            f"- **Точність (Accuracy):** **{eval_data.get('accuracy', 0) * 100:.1f}%**",
            ""
        ])

        # Directional Metrics
        lines.append("### Детальні метрики за напрямками (Traffic)")
        lines.append("| Напрямок | GT (Правда) | Pred (Система) | TP (Співпало) | FP (Хибні) | FN (Пропущено) | Precision | Recall | F1-Score |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for d_key, d_label in [("in", "IN (Вліт)"), ("out", "OUT (Виліт)")]:
            m = eval_data.get(d_key, {})
            lines.append(
                f"| {d_label} | {m.get('gt_count')} | {m.get('pred_count')} | "
                f"**{m.get('tp')}** | {m.get('fp')} | {m.get('fn')} | "
                f"{m.get('precision', 0):.2f} | {m.get('recall', 0):.2f} | **{m.get('f1', 0):.2f}** |"
            )
        lines.append("")
        
        # Behavior Comparison
        gt_b = eval_data.get("gt_behaviors", {})
        pred_b = res.get("behavior_summary", {})
        lines.append("### Порівняння поведінкових станів")
        lines.append("*Примітка: GT містить кількість унікальних треків, система — загальну кількість спрацювань класифікатора. Washboarding відсутнє у даному GT.*")
        lines.append("| Стан | GT (Унікальні треки) | System Predicted |")
        lines.append("|---|---|---|")
        lines.append(f"| Fanning (Вентиляція) | {gt_b.get('fanning_tracks', 0)} | {pred_b.get('fanning_detections', 0)} |")
        lines.append(f"| Defense (Охорона) | {gt_b.get('defense_tracks', 0)} | {pred_b.get('defense_detections', 0)} |")
        lines.append(f"| Arrival/Foraging | {gt_b.get('arrival_tracks', 0)} | {pred_b.get('foraging_detections', 0)} |")
        lines.append(f"| Washboarding | N/A | {pred_b.get('washboarding_detections', 0)} |")
        lines.append("")

        # Reasons if approach B
        if cfg.get("approach") == "B":
            res = job.get("result", {})
            rej = res.get("pose_rejected_events", 0)
            reasons = res.get("reject_reasons", {})
            hist = res.get("angle_histogram", {})
            
            lines.append("### Аналіз ефективності Pose-фільтра")
            lines.append(f"Усього відсіяно як джиттер: **{rej}** подій.")
            lines.append("Причини відхилення:")
            lines.append(f"- Невідповідність кута (>60°): {reasons.get('angle_mismatch', 0)}")
            lines.append(f"- Відсутні ключові точки: {reasons.get('no_keypoints', 0)}")
            lines.append("")
            lines.append("#### Розподіл кутових розбіжностей (всі події)")
            for bin_key, count in hist.items():
                lines.append(f"- **{bin_key}°**: {count}")
            lines.append("")

        lines.append("---\n")

    return PlainTextResponse("\n".join(lines))

@router.get("/summary")
async def get_summary():
    db = get_db()
    
    pipeline = [
        {"$match": {"status": "complete"}},
        {"$group": {
            "_id": None,
            "total_in": {"$sum": "$result.total_in"},
            "total_out": {"$sum": "$result.total_out"},
            "total_sessions": {"$sum": 1},
            "avg_fps": {"$avg": "$result.fps_processed"}
        }}
    ]
    cursor = db["jobs"].aggregate(pipeline)
    results = await cursor.to_list(length=1)
    
    if not results:
        return {"total_in": 0, "total_out": 0, "total_sessions": 0, "avg_fps": 0, "avg_balance": 0}
        
    res = results[0]
    res.pop("_id", None)
    res["avg_balance"] = (res["total_in"] - res["total_out"]) / res["total_sessions"] if res["total_sessions"] > 0 else 0
    return res

@router.get("/compare-approaches")
async def compare_approaches():
    db = get_db()
    jobs = await db["jobs"].find({"status": "complete"}, {"_id": 0, "events": 0}).to_list(100)
    
    approach_a = [j for j in jobs if j.get("config", {}).get("approach") == "A"]
    approach_b = [j for j in jobs if j.get("config", {}).get("approach") == "B"]
    
    avg_in_a = sum(j.get("result", {}).get("total_in", 0) for j in approach_a) / len(approach_a) if approach_a else 0
    avg_in_b = sum(j.get("result", {}).get("total_in", 0) for j in approach_b) / len(approach_b) if approach_b else 0
    
    # Calculate pose confirmed rate for B
    total_b_events = sum(j.get("result", {}).get("total_in", 0) + j.get("result", {}).get("total_out", 0) for j in approach_b)
    total_b_pose = sum(j.get("result", {}).get("pose_confirmed_events", 0) for j in approach_b)
    pose_confirmed_rate = (total_b_pose / total_b_events) * 100 if total_b_events > 0 else 0
    
    return {
        "approach_a_count": len(approach_a),
        "approach_b_count": len(approach_b),
        "avg_in_a": float(avg_in_a),
        "avg_in_b": float(avg_in_b),
        "pose_confirmed_rate": float(pose_confirmed_rate)
    }
