from fastapi import APIRouter
from db.mongodb import get_db

router = APIRouter(prefix="/analytics", tags=["Analytics"])

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
