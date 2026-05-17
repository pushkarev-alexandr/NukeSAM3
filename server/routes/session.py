from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..models import SessionCreateRequest, SessionCreateResponse, SessionDeleteResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/session/create", response_model=SessionCreateResponse)
async def create_session(body: SessionCreateRequest, request: Request) -> SessionCreateResponse:
    app = request.app
    video_path = body.video_path

    if not Path(video_path).exists():
        raise HTTPException(status_code=400, detail=f"video_path does not exist: {video_path}")

    # Start a SAM3 video predictor session on the GPU thread
    try:
        result = await app.state.gpu_worker.submit(
            lambda: _start_video_session(app.state.video_predictor, video_path),
            priority=0,
        )
    except Exception as exc:
        logger.exception("Failed to start SAM3 video session")
        raise HTTPException(status_code=500, detail=str(exc))

    sam3_video_session_id = result["session_id"]
    frame_count = result["frame_count"]
    width = result["width"]
    height = result["height"]

    session = app.state.sessions.create(
        video_path=video_path,
        output_dir=body.output_dir,
        client_id=body.client_id,
        frame_count=frame_count,
        width=width,
        height=height,
        sam3_video_session_id=sam3_video_session_id,
    )

    from ..config import config
    return SessionCreateResponse(
        session_id=session.session_id,
        frame_count=frame_count,
        width=width,
        height=height,
        expires_in_seconds=config.SESSION_TTL_SECONDS,
    )


@router.delete("/session/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, request: Request) -> SessionDeleteResponse:
    app = request.app
    try:
        session = app.state.sessions.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    session.request_cancel()

    import torch
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    predictor = app.state.video_predictor
    vid_sid = session.sam3_video_session_id
    if vid_sid:
        await app.state.gpu_worker.submit(
            lambda: predictor.handle_request({"type": "close_session", "session_id": vid_sid}),
            priority=0,
        )

    app.state.sessions.delete(session_id)

    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    freed_gb = max(0, (before - after)) / 1e9

    return SessionDeleteResponse(is_success=True, gpu_memory_freed_gb=round(freed_gb, 3))


# ------------------------------------------------------------------
# Helpers (run inside GPU thread via lambda)
# ------------------------------------------------------------------

def _start_video_session(predictor, video_path: str) -> dict:
    response = predictor.handle_request({
        "type": "start_session",
        "resource_path": video_path,
    })
    inference_state = predictor._all_inference_states[response["session_id"]]["state"]
    return {
        "session_id": response["session_id"],
        "frame_count": inference_state["num_frames"],
        "width": inference_state["orig_width"],
        "height": inference_state["orig_height"],
    }
