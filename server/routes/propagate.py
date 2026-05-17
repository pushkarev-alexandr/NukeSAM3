"""
POST /infer/propagate  →  SSE stream

Uses the SAM3 video predictor's propagate_in_video generator.
Each frame result is pushed as an SSE 'progress' event.
Sends 'complete' or 'cancelled' when done.

The GPU thread runs the generator; an asyncio.Queue bridges it to the SSE stream.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..mask_io import write_mask_exr
from ..models import PropagateRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/infer/propagate")
async def propagate(body: PropagateRequest, request: Request):
    app = request.app

    try:
        session = app.state.sessions.get(body.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    # Make sure there's a prompt set on the video predictor for this session
    # (caller should have used /infer/add_prompt first)

    output_dir = Path(body.output_dir) if body.output_dir else session.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    return StreamingResponse(
        _sse_generator(app, session, body, output_dir),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------
# SSE generator
# ------------------------------------------------------------------

async def _sse_generator(app, session, body: PropagateRequest, output_dir: Path):
    loop = asyncio.get_event_loop()
    event_q: asyncio.Queue = asyncio.Queue()

    session.reset_cancel()
    session.propagation_running = True

    # Run GPU work in a thread so we don't block the event loop
    thread = threading.Thread(
        target=_propagate_worker,
        args=(app.state.video_predictor, session, body, output_dir, event_q, loop),
        daemon=True,
    )
    thread.start()

    try:
        while True:
            event = await event_q.get()
            if event is None:  # sentinel: worker finished
                break
            yield event
    finally:
        session.propagation_running = False


def _propagate_worker(predictor, session, body: PropagateRequest, output_dir: Path, q: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Runs in a background thread; pushes SSE-formatted strings into q."""

    def push(event_type: str, data: dict):
        line = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        loop.call_soon_threadsafe(q.put_nowait, line)

    total = session.frame_count
    written = 0
    t_start = time.perf_counter()

    try:
        stream_req = {
            "type": "propagate_in_video",
            "session_id": session.sam3_video_session_id,
            "propagation_direction": body.propagation_direction,
        }

        for frame_response in predictor.handle_stream_request(stream_req):
            if session.propagation_cancel.is_set():
                push("cancelled", {"frames_written": written, "reason": "interactive_preempt"})
                return

            frame_idx = frame_response["frame_index"]
            outputs = frame_response["outputs"]
            masks = outputs.get("out_binary_masks")

            if masks is not None:
                if not isinstance(masks, np.ndarray):
                    masks = np.asarray(masks, dtype=bool)
                exr_path = output_dir / body.frame_filename_pattern % frame_idx
                write_mask_exr(exr_path, masks)
                written += 1
            else:
                exr_path = output_dir / body.frame_filename_pattern % frame_idx

            elapsed_ms = int((time.perf_counter() - t_start) * 1000)
            pct = int((written / max(total, 1)) * 100)
            push("progress", {
                "frame_index": frame_idx,
                "total_frames": total,
                "percent": pct,
                "exr_path": str(exr_path),
                "elapsed_ms": elapsed_ms,
            })

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        push("complete", {
            "total_frames_written": written,
            "output_dir": str(output_dir),
            "duration_ms": elapsed_ms,
        })

    except Exception as exc:
        logger.exception("Propagation error")
        push("error", {"message": str(exc), "frames_written": written})
    finally:
        # Sentinel to stop the async generator
        loop.call_soon_threadsafe(q.put_nowait, None)


# ------------------------------------------------------------------
# Add prompt endpoint (populates video predictor before propagation)
# ------------------------------------------------------------------

from pydantic import BaseModel as _BaseModel
from typing import Optional as _Optional


class AddPromptRequest(_BaseModel):
    session_id: str
    frame_index: int = 0
    text: _Optional[str] = None
    points: _Optional[list[list[float]]] = None
    point_labels: _Optional[list[int]] = None
    bounding_boxes: _Optional[list[list[float]]] = None
    bounding_box_labels: _Optional[list[int]] = None
    obj_id: _Optional[int] = None
    clear_old_points: bool = True
    clear_old_boxes: bool = True


@router.post("/infer/add_prompt")
async def add_prompt(body: AddPromptRequest, request: Request):
    app = request.app
    try:
        session = app.state.sessions.get(body.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    req = {
        "type": "add_prompt",
        "session_id": session.sam3_video_session_id,
        "frame_index": body.frame_index,
        "text": body.text,
        "points": body.points,
        "point_labels": body.point_labels,
        "bounding_boxes": body.bounding_boxes,
        "bounding_box_labels": body.bounding_box_labels,
        "clear_old_points": body.clear_old_points,
        "clear_old_boxes": body.clear_old_boxes,
    }
    if body.obj_id is not None:
        req["obj_id"] = body.obj_id

    try:
        result = await app.state.gpu_worker.submit(
            lambda: app.state.video_predictor.handle_request(req),
            priority=0,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"frame_index": body.frame_index, "ok": True}


@router.post("/session/{session_id}/reset")
async def reset_session(session_id: str, request: Request):
    app = request.app
    try:
        session = app.state.sessions.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    await app.state.gpu_worker.submit(
        lambda: app.state.video_predictor.handle_request({
            "type": "reset_session",
            "session_id": session.sam3_video_session_id,
        }),
        priority=0,
    )
    session.image_state = None
    return {"is_success": True}
