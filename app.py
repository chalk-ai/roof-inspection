"""Drone roof inspection for claims: sample a roof video, find damage frame by frame, summarize.

Runs in the CPU scaling group that deploy.py creates. The vision-language model is a separate GPU
scaling group serving Qwen3-VL-8B-Instruct through vLLM's OpenAI-compatible API (VLM_URL).

HTTP API:
  GET  /                    the inspection page (index.html)
  GET  /api/status          model name and the bundled sample videos
  GET  /samples/{name}.mp4  a bundled sample video
  POST /api/analyze         ?sample=<name>, or a multipart `video` upload. Streams NDJSON:
                            {"type": "start"}, one {"type": "frame"} per keyframe with findings
                            and boxes, then {"type": "report"} and {"type": "done"}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

# request.form() yields Starlette's UploadFile; FastAPI's is a subclass, so isinstance needs this one.
from starlette.datastructures import UploadFile

VLM_URL = os.environ["VLM_URL"].rstrip("/")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", "/samples"))
PORT = int(os.environ.get("PORT", "8080"))
INDEX_HTML = Path(__file__).with_name("index.html")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_DURATION_S = 300.0
MAX_FRAMES = 14
MIN_FRAME_SPACING_S = 1.5
FRAME_WIDTH = 896
FRAME_CONCURRENCY = 6
# Whole analyses at once, across all visitors; the rest get a "busy" answer instead of queueing.
MAX_ACTIVE_ANALYSES = 3

SAMPLES = {
    "ian": {
        "title": "Single home, Fort Myers, FL",
        "event": "Hurricane Ian, October 2022",
        "credit": "U.S. Army Corps of Engineers, Jacksonville District (public domain)",
    },
    "ida": {
        "title": "Residential street, LaPlace, LA",
        "event": "Hurricane Ida, September 2021",
        "credit": "U.S. Army Corps of Engineers (public domain)",
    },
}

FINDING_TYPES = [
    "missing_shingles",
    "exposed_decking",
    "temporary_tarp",
    "structural_damage",
    "debris_on_roof",
    "damaged_flashing_or_vents",
    "fallen_tree_or_limb",
    "gutter_or_fascia_damage",
    "debris_on_ground",
    "other",
]

FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string", "description": "One short sentence: what this frame shows."},
        "roofs_visible": {"type": "integer", "minimum": 0, "maximum": 50},
        "frame_quality": {"type": "string", "enum": ["clear", "partial", "unusable"]},
        "findings": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": FINDING_TYPES},
                    "severity": {"type": "string", "enum": ["minor", "moderate", "severe"]},
                    "description": {"type": "string", "maxLength": 160},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["type", "severity", "description", "bbox"],
            },
        },
    },
    "required": ["scene", "roofs_visible", "frame_quality", "findings"],
}

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "maxLength": 140},
        "property_overview": {"type": "string", "maxLength": 400},
        "roof_type": {"type": "string", "maxLength": 80},
        "overall_severity": {"type": "string", "enum": ["none", "minor", "moderate", "severe"]},
        "estimated_damaged_area_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        "recommendation": {
            "type": "string",
            "enum": ["no_action", "repair", "emergency_tarp", "partial_replacement", "full_replacement"],
        },
        "recommendation_reason": {"type": "string", "maxLength": 400},
        "damage_items": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": FINDING_TYPES},
                    "severity": {"type": "string", "enum": ["minor", "moderate", "severe"]},
                    "evidence_frames": {"type": "array", "items": {"type": "integer"}, "maxItems": 6},
                    "notes": {"type": "string", "maxLength": 240},
                },
                "required": ["category", "severity", "evidence_frames", "notes"],
            },
        },
        "verify_on_site": {"type": "array", "items": {"type": "string", "maxLength": 160}, "maxItems": 6},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "headline",
        "property_overview",
        "roof_type",
        "overall_severity",
        "estimated_damaged_area_pct",
        "recommendation",
        "recommendation_reason",
        "damage_items",
        "verify_on_site",
        "confidence",
    ],
}

FRAME_PROMPT = """You are assisting a property insurance claims adjuster reviewing drone footage of roofs after a storm.
Inspect this frame and fill in every field:
- scene: one plain-English sentence describing the view, for example "Low overhead view of a single-story
  house whose shingle roof has a large section of missing shingles near the ridge."
- roofs_visible: how many roofs are at least partly visible in the frame.
- frame_quality: clear, partial (roof cut off or obstructed), or unusable (mostly sky, ground or blur).
- findings: only damage you can actually see, each with a tight bounding box [x1, y1, x2, y2] in relative
  coordinates from 0 to 1000 over this image. Missing shingles show as darker or differently colored patches
  in the shingle pattern; exposed decking shows as bare plywood. Blue or grey plastic sheeting on a roof is a
  temporary tarp: report it as temporary_tarp, since it marks damage underneath.
Do not report undamaged roofs. If there is no visible damage, return an empty findings list."""

REPORT_PROMPT = """You are assisting a property insurance claims adjuster. Below are per-frame findings from an
AI review of drone footage of {subject}, plus a few of the frames themselves (labelled with their frame numbers).
Write a concise pre-inspection damage summary for the adjuster. Base every statement on the findings and frames;
reference evidence by frame number. Estimate the damaged share of the visible roof area. List what the adjuster
should verify on site, because footage cannot show everything (for example interior water intrusion or decking
condition under tarps). A temporary tarp is not a fix: when one is already in place, the roof still needs a permanent
remedy, so recommend repair, partial_replacement or full_replacement (never no_action or another tarp).
Choose replacement over repair when damage spans much of a roof plane. This is decision support, not a coverage determination.
Write every field in plain English.

Per-frame findings (JSON):
{findings}"""

log = logging.getLogger("roof")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# --- video ----------------------------------------------------------------------------------


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def keyframe_times(duration: float) -> list[float]:
    """Evenly spaced sample times, at most MAX_FRAMES and at least MIN_FRAME_SPACING_S apart."""
    count = max(1, min(MAX_FRAMES, int(duration / MIN_FRAME_SPACING_S)))
    step = duration / count
    return [round(step * (i + 0.5), 2) for i in range(count)]


def extract_frame(path: Path, t: float) -> bytes:
    """One JPEG at time t, scaled to FRAME_WIDTH. Seeking before -i keeps this fast on long videos."""
    out = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
            "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "4", "-f", "image2", "pipe:1",
        ],
        capture_output=True,
        check=True,
    )  # fmt: skip
    if not out.stdout:
        raise ValueError(f"no frame at {t}s")
    return out.stdout


# --- model ----------------------------------------------------------------------------------


def _auth_headers() -> dict[str, str]:
    # The model group is authenticated; this replica's workload identity token rotates on disk.
    path = os.environ.get("CHALK_WEB_IDENTITY_TOKEN_FILE")
    return {"Authorization": f"Bearer {Path(path).read_text().strip()}"} if path else {}


def _image_part(jpeg: bytes) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()},
    }


async def chat_json(
    client: httpx.AsyncClient, content: list[dict[str, Any]], schema: dict[str, Any], max_tokens: int
) -> dict[str, Any]:
    r = await client.post(
        f"{VLM_URL}/v1/chat/completions",
        headers=_auth_headers(),
        json={
            "model": VLM_MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema},
            },
        },
    )
    if r.status_code >= 400:
        raise RuntimeError(f"model returned {r.status_code}: {r.text[:300]}")
    choice = r.json()["choices"][0]
    try:
        return json.loads(choice["message"]["content"])
    except json.JSONDecodeError as exc:
        log.warning(
            "unparseable model output (finish=%s): %r", choice.get("finish_reason"), choice["message"]["content"][:400]
        )
        raise RuntimeError(f"model output did not parse (finish_reason={choice.get('finish_reason')})") from exc


def _clean_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Orders each box as (left, top, right, bottom) and drops degenerate ones."""
    out = []
    for f in findings:
        x1, y1, x2, y2 = f["bbox"]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            out.append({**f, "bbox": [x1, y1, x2, y2]})
    return out


SEVERITY_RANK = {"minor": 1, "moderate": 2, "severe": 3}


# --- app ------------------------------------------------------------------------------------

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    state["client"] = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0))
    state["active"] = asyncio.Semaphore(MAX_ACTIVE_ANALYSES)
    yield
    await state["client"].aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/api/status")
async def status() -> JSONResponse:
    samples = {k: v for k, v in SAMPLES.items() if (SAMPLES_DIR / f"{k}.mp4").exists()}
    return JSONResponse({"model": VLM_MODEL, "samples": samples})


@app.get("/samples/{name}.mp4")
async def sample_video(name: str) -> FileResponse:
    if name not in SAMPLES:
        raise HTTPException(404)
    return FileResponse(SAMPLES_DIR / f"{name}.mp4", media_type="video/mp4")


async def analyze(path: Path, subject: str) -> AsyncIterator[dict[str, Any]]:
    client: httpx.AsyncClient = state["client"]
    t_start = time.perf_counter()
    duration = await asyncio.to_thread(probe_duration, path)
    if duration > MAX_DURATION_S:
        raise ValueError(f"video is {duration:.0f} s; the limit is {MAX_DURATION_S:.0f} s")
    times = keyframe_times(duration)
    yield {"type": "start", "duration": round(duration, 2), "frames": len(times), "model": VLM_MODEL}

    slots = asyncio.Semaphore(FRAME_CONCURRENCY)
    frames: dict[int, bytes] = {}

    async def one(i: int, t: float) -> dict[str, Any]:
        async with slots:
            jpeg = await asyncio.to_thread(extract_frame, path, t)
            frames[i] = jpeg
            t0 = time.perf_counter()
            content = [_image_part(jpeg), {"type": "text", "text": FRAME_PROMPT}]
            # One retry, then give up on this frame alone: a single bad answer should cost a
            # keyframe, not the whole inspection.
            for attempt in range(2):
                try:
                    result = await chat_json(client, content, FRAME_SCHEMA, 1200)
                    break
                except RuntimeError as exc:
                    if attempt == 1:
                        return {"type": "frame", "i": i, "t": t, "image": base64.b64encode(jpeg).decode(),
                                "scene": f"Skipped: {exc}", "roofs_visible": 0, "frame_quality": "unusable",
                                "findings": [], "ms": round((time.perf_counter() - t0) * 1000)}  # fmt: skip
            return {
                "type": "frame",
                "i": i,
                "t": t,
                "image": base64.b64encode(jpeg).decode(),
                "scene": result["scene"],
                "roofs_visible": result["roofs_visible"],
                "frame_quality": result["frame_quality"],
                "findings": _clean_findings(result["findings"]),
                "ms": round((time.perf_counter() - t0) * 1000),
            }

    per_frame: list[dict[str, Any]] = []
    for task in asyncio.as_completed([one(i, t) for i, t in enumerate(times)]):
        event = await task
        per_frame.append(event)
        yield event

    per_frame.sort(key=lambda e: e["i"])
    compact = [
        {"frame": e["i"], "t": e["t"], "scene": e["scene"], "quality": e["frame_quality"],
         "findings": [{k: f[k] for k in ("type", "severity", "description")} for f in e["findings"]]}
        for e in per_frame
    ]  # fmt: skip
    # Show the model the frames with the most severe evidence, so the summary is grounded in pixels.
    ranked = sorted(
        per_frame,
        key=lambda e: max((SEVERITY_RANK[f["severity"]] for f in e["findings"]), default=0) * 10 + len(e["findings"]),
        reverse=True,
    )[:4]
    content: list[dict[str, Any]] = []
    for e in sorted(ranked, key=lambda e: e["i"]):
        content += [{"type": "text", "text": f"Frame {e['i']}:"}, _image_part(frames[e["i"]])]
    content.append({"type": "text", "text": REPORT_PROMPT.format(subject=subject, findings=json.dumps(compact))})
    t0 = time.perf_counter()
    report = await chat_json(client, content, REPORT_SCHEMA, 1400)
    yield {"type": "report", "report": report, "ms": round((time.perf_counter() - t0) * 1000)}
    yield {"type": "done", "total_s": round(time.perf_counter() - t_start, 1)}


def _stream(path: Path, subject: str, cleanup: Path | None) -> StreamingResponse:
    async def body() -> AsyncIterator[str]:
        sem: asyncio.Semaphore = state["active"]
        if sem.locked():
            yield json.dumps({"type": "busy"}) + "\n"
            return
        async with sem:
            try:
                async for event in analyze(path, subject):
                    yield json.dumps(event, separators=(",", ":")) + "\n"
            except Exception as exc:
                log.exception("analysis failed")
                yield json.dumps({"type": "error", "message": str(exc)[:300]}) + "\n"
            finally:
                if cleanup is not None:
                    shutil.rmtree(cleanup, ignore_errors=True)

    return StreamingResponse(body(), media_type="application/x-ndjson")


@app.post("/api/analyze")
async def analyze_endpoint(request: Request, sample: str | None = None) -> StreamingResponse:
    if sample is not None:
        if sample not in SAMPLES:
            raise HTTPException(404, "unknown sample")
        return _stream(SAMPLES_DIR / f"{sample}.mp4", SAMPLES[sample]["title"], None)

    form = await request.form(max_part_size=MAX_UPLOAD_BYTES)
    video = form.get("video")
    if not isinstance(video, UploadFile):
        raise HTTPException(400, "send ?sample=<name> or a multipart 'video' file")
    tmp = Path(tempfile.mkdtemp(prefix="roof-"))
    dest = tmp / "upload"
    size = 0
    with dest.open("wb") as out:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                shutil.rmtree(tmp, ignore_errors=True)
                raise HTTPException(413, "video is larger than 200 MB")
            out.write(chunk)
    return _stream(dest, "the uploaded property", tmp)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
