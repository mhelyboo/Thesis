from urllib import request

from ultralytics import YOLO
from keras_facenet import FaceNet
import cv2, numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytz
from fastapi import FastAPI, Depends, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from database import get_db

PH_TZ = pytz.timezone("Asia/Manila")

app = FastAPI(title="GSU FaceTend API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
@app.get("/ping")
async def ping():
    return {"status": "ok"}

MEDIA_DIR = Path("media/faces")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


yolo_model = YOLO(r"C:\Users\mhelr\AppData\Local\Programs\Python\Python311\FaceTend\best_faces.pt")
facenet = FaceNet()

# Single global camera
camera = cv2.VideoCapture(1)
if not camera.isOpened():
    raise RuntimeError("Camera not opened")


def get_face_embedding(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    embeddings = facenet.embeddings([img_rgb])
    return embeddings[0]


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def get_frame() -> np.ndarray:
    if not camera.isOpened():
        raise RuntimeError("Camera not opened")
    ok, frame = camera.read()
    if not ok or frame is None:
        raise RuntimeError("Cannot read frame from camera")
    return frame


def gen_yolo_frames():
    while True:
        frame = get_frame()

        results = yolo_model(frame, conf=0.3)
        r = results[0]

        annotated = frame.copy()
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.get("/video-stream")
async def video_stream():
    return StreamingResponse(
        gen_yolo_frames(),   # now /video-stream shows boxes
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def gen_dashboard_frames():
    while True:
        try:
            frame = get_frame()
        except Exception as e:
            print("Stream error:", e)
            break

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.get("/register", response_class=HTMLResponse)
async def show_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register-capture")
@app.post("/register-capture")
async def register_capture(
    student_id: str = Form(...),
    name: str = Form(...),
    db=Depends(get_db),
):
    student = db.students.find_one({"student_id": student_id}) or {}
    existing_paths = student.get("face_images", [])
    current_idx = len(existing_paths)

    if current_idx >= 10:
        return {"status": "done", "count": current_idx}

    frame = get_frame()

    results = yolo_model(frame, conf=0.3)
    r = results[0]
    if len(r.boxes) == 0:
        return {"status": "no_face", "count": current_idx}

    areas = []
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        areas.append(((x2 - x1) * (y2 - y1), (x1, y1, x2, y2)))

    if not areas:
        return {"status": "no_face", "count": current_idx}

    _, (x1, y1, x2, y2) = max(areas)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"status": "no_face", "count": current_idx}

@app.post("/recognize-face")
async def recognize_face(
    file: UploadFile = File(...),
    db=Depends(get_db),
):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image.")

    cv2.imwrite("debug_recognize.jpg", img)

    results = yolo_model(img, conf=0.1)
    r = results[0]
    print("recognize boxes:", len(r.boxes))

    if len(r.boxes) == 0:
        raise HTTPException(status_code=400, detail="No object detected.")

    candidates = list(db.students.find({"embedding": {"$ne": None}}))

    recognized = []
    now_utc = datetime.now(timezone.utc)

    THRESH = 0.9
    COOLDOWN_MINUTES = 10

    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        emb = get_face_embedding(crop)

        best_id = None
        best_name = None
        best_dist = 1e9

        for s in candidates:
            ref_emb = np.array(s["embedding"], dtype=np.float32)
            d = l2_distance(emb, ref_emb)
            if d < best_dist:
                best_dist = d
                best_id = s["student_id"]
                best_name = s["name"]

        print("BEST:", best_name, best_dist)

        if best_dist > THRESH or best_id is None:
            # log Unknown
            doc = {
                "student_id": None,
                "name": "Unknown",
                "timestamp": now_utc,
                "status": "unknown",
                "distance": float(best_dist),
                "logout_time": None,
            }
            db.attendance.insert_one(doc)
            recognized.append(
                {
                    "student_id": None,
                    "name": "Unknown",
                    "status": "unknown",
                    "distance": float(best_dist),
                }
            )
            continue

        last_log = db.attendance.find_one(
            {"student_id": best_id},
            sort=[("timestamp", -1)],
        )

        if last_log:
            last_time = last_log["timestamp"]
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            time_diff = now_utc - last_time
            if time_diff < timedelta(minutes=COOLDOWN_MINUTES):
                continue

        new_status = (
            "in" if not last_log or last_log.get("status") == "out" else "out"
        )

        doc = {
            "student_id": best_id,
            "name": best_name,
            "timestamp": now_utc,
            "status": new_status,
            "distance": float(best_dist),
            "logout_time": None if new_status == "in" else now_utc,
        }
        db.attendance.insert_one(doc)
        recognized.append(
            {
                "student_id": best_id,
                "name": best_name,
                "status": new_status,
                "distance": float(best_dist),
            }
        )

    return recognized


ADMIN_USER = {
    "name": "Mhelrhose Hervias",
    "role": "Project Manager",
}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db=Depends(get_db)):
    now_utc = datetime.now(timezone.utc)
    start_of_day_utc = datetime(
        now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc
    )

    present_count = db.attendance.count_documents(
        {
            "timestamp": {"$gte": start_of_day_utc},
            "status": "in",
            "logout_time": None,
        }
    )
    total_students = db.students.count_documents({})
    absent_count = max(total_students - present_count, 0)

    attendance_today = []
    cursor = (
        db.attendance.find({"timestamp": {"$gte": start_of_day_utc}})
        .sort("timestamp", -1)
    )
    for doc in cursor:
        ts_utc = doc["timestamp"]
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)
        ts_ph = ts_utc.astimezone(PH_TZ)
        attendance_today.append(
            {
                "student_id": doc.get("student_id"),
                "name": doc.get("name"),
                "status": doc.get("status"),
                "time_in": ts_ph.strftime("%H:%M:%S"),
            }
        )

    recent_recognitions = []
    THRESH = 1.5
    cursor_recent = db.attendance.find().sort("timestamp", -1).limit(5)
    for doc in cursor_recent:
        ts_utc = doc["timestamp"]
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    ts_ph = ts_utc.astimezone(PH_TZ)

    if doc.get("status") == "unknown":
        name = "Unknown"
        conf = 0.0
    else:
        name = doc.get("name")
        dist = float(doc.get("distance", 0))
        conf = max(0.0, min(100.0, 100 * (1.0 - dist / THRESH)))

    recent_recognitions.append(
        {
            "name": name,
            "student_id": doc.get("student_id"),
            "confidence": round(conf, 1),
            "time": ts_ph.strftime("%I:%M:%S %p"),
        }
    )

    stats = {
        "total_students": total_students,
        "present": present_count,
        "absent": absent_count,
        "accuracy": 0,
        "speed_ms": 0,
    }

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "stats": stats,
            "attendance_today": attendance_today,
            "recent_recognitions": recent_recognitions,
            "admin": ADMIN_USER,
        },
    )

@app.get("/register", response_class=HTMLResponse)
async def show_register(request: Request):
    return templates.TemplateResponse(
        "register.html",
        {
            "request": request,
            "admin": ADMIN_USER,
        },
    )


@app.get("/api/attendance-today")
async def api_attendance_today(db=Depends(get_db)):
    now_utc = datetime.now(timezone.utc)
    start_of_day_utc = datetime(
        now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc
    )

    rows = []
    cursor = (
        db.attendance.find({"timestamp": {"$gte": start_of_day_utc}})
        .sort("timestamp", -1)
    )
    for doc in cursor:
        ts_utc = doc["timestamp"]
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)
        ts_ph = ts_utc.astimezone(PH_TZ)

        logout_time = None
        if doc.get("logout_time"):
            logout_utc = doc["logout_time"]
            if logout_utc.tzinfo is None:
                logout_utc = logout_utc.replace(tzinfo=timezone.utc)
            logout_time = logout_utc.astimezone(PH_TZ).strftime("%H:%M:%S")

        rows.append(
            {
                "student_id": doc.get("student_id"),
                "name": doc.get("name"),
                "status": doc.get("status"),
                "time_in": ts_ph.strftime("%H:%M:%S"),
                "time_out": logout_time,
            }
        )
    return rows


def handle_detection(db, student_id, name, camera: str = "entrance"):
    now_utc = datetime.now(timezone.utc)
    if camera == "entrance":
        existing = db.attendance.find_one(
            {"student_id": student_id, "status": "in", "logout_time": None}
        )
        if not existing:
            db.attendance.insert_one(
                {
                    "student_id": student_id,
                    "name": name,
                    "status": "in",
                    "timestamp": now_utc,
                    "logout_time": None,
                }
            )
    elif camera == "exit":
        existing = db.attendance.find_one(
            {"student_id": student_id, "status": "in", "logout_time": None}
        )
        if existing:
            db.attendance.update_one(
                {"_id": existing["_id"]},
                {"$set": {"logout_time": now_utc, "status": "out"}},
            )


@app.get("/simulate-entrance")
async def simulate_entrance(db=Depends(get_db)):
    student_id = "12345"
    name = "Test Student"
    handle_detection(db, student_id, name, camera="entrance")
    return {"message": f"{name} logged in at entrance"}


@app.get("/simulate-exit")
async def simulate_exit(db=Depends(get_db)):
    student_id = "12345"
    name = "Test Student"
    handle_detection(db, student_id, name, camera="exit")
    return {"message": f"{name} logged out at exit"}


def compute_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    mean_score = (precision + recall + f1) / 3.0
    return accuracy, precision, recall, f1, mean_score


@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation(request: Request, db=Depends(get_db)):
    metrics_doc = db.metrics.find_one({"name": "model_eval"}) or {}
    tp = metrics_doc.get("tp", 0)
    fp = metrics_doc.get("fp", 0)
    fn = metrics_doc.get("fn", 0)
    tn = metrics_doc.get("tn", 0)
    acc, prec, rec, f1, mean_score = compute_metrics(tp, fp, fn, tn)

    metrics = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "mean_score": round(mean_score, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": tp + fp + fn + tn,
        "threshold": metrics_doc.get("threshold", 1.5),
    }

    return templates.TemplateResponse(
        "evaluation.html",
        {"request": request, "metrics": metrics, "per_class": [], "admin": ADMIN_USER},
    )
