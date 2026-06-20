"""
FastAPI バックエンド — 日本語→英語 自動吹き替えシステム Pro
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.pipeline import run_transcription, run_pipeline, update_status

app = FastAPI(title="日本語→英語 自動吹き替えシステム Pro")

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


# ── ページ配信 ────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse("app/static/index.html")


@app.get("/edit")
async def edit_page():
    return FileResponse("app/static/edit.html")


# ── 動画アップロード ──────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(default="large-v3"),
):
    job_id  = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()

    suffix     = Path(file.filename).suffix or ".mp4"
    video_path = job_dir / f"original{suffix}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    update_status(job_id, "uploaded", 0, "アップロード完了。文字起こしを開始します...")
    background_tasks.add_task(run_transcription, job_id, str(video_path), model)

    return {"job_id": job_id}


# ── ジョブ状態確認 ────────────────────────────────────────────────────
@app.get("/jobs/{job_id}/status")
async def get_status(job_id: str):
    path = JOBS_DIR / job_id / "status.json"
    if not path.exists():
        return JSONResponse({"error": "ジョブが見つかりません"}, status_code=404)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 日本語セグメント取得（編集済み優先） ─────────────────────────────
@app.get("/jobs/{job_id}/segments")
async def get_segments(job_id: str):
    job_dir = JOBS_DIR / job_id
    for name in ["segments_ja_edited.json", "segments_ja.json"]:
        path = job_dir / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return JSONResponse({"error": "セグメントが見つかりません"}, status_code=404)


# ── 日本語セグメント保存 ──────────────────────────────────────────────
class Segment(BaseModel):
    start: float
    end:   float
    text:  str


@app.put("/jobs/{job_id}/segments")
async def save_segments(job_id: str, segments: List[Segment] = Body(...)):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        return JSONResponse({"error": "ジョブが見つかりません"}, status_code=404)

    path = job_dir / "segments_ja_edited.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in segments], f, ensure_ascii=False, indent=2)

    return {"saved": True, "count": len(segments)}


# ── 処理実行（翻訳→TTS→動画合成） ───────────────────────────────────
@app.post("/jobs/{job_id}/run")
async def run_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    voice: str = Form(default="female"),
    subtitle: bool = Form(default=True),
    subtitle_lang: str = Form(default="en"),
):
    if not (JOBS_DIR / job_id).exists():
        return JSONResponse({"error": "ジョブが見つかりません"}, status_code=404)

    update_status(job_id, "processing", 0, "処理を開始しています...")
    background_tasks.add_task(run_pipeline, job_id, voice, subtitle, subtitle_lang)

    return {"started": True}


# ── 完成動画ダウンロード ──────────────────────────────────────────────
@app.get("/jobs/{job_id}/download")
async def download_video(job_id: str):
    output_path = JOBS_DIR / job_id / "output.mp4"
    if not output_path.exists():
        return JSONResponse({"error": "出力動画がまだ完成していません"}, status_code=404)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="output_english.mp4",
    )


# ── 字幕ダウンロード ──────────────────────────────────────────────────
@app.get("/jobs/{job_id}/subtitle")
async def download_subtitle(job_id: str):
    path = JOBS_DIR / job_id / "subtitle.srt"
    if not path.exists():
        return JSONResponse({"error": "字幕ファイルがまだ完成していません"}, status_code=404)
    return FileResponse(path, media_type="text/plain; charset=utf-8",
                        filename="subtitle_english.srt")
