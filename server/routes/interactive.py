"""
POST /infer/interactive

Runs single-frame inference using Sam3Processor (image model).
Priority 0 — always preempts propagation on the same session.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from ..mask_io import masks_to_png_b64_list, write_mask_exr
from ..models import InteractiveRequest, InteractiveResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/infer/interactive", response_model=InteractiveResponse)
async def interactive_infer(body: InteractiveRequest, request: Request) -> InteractiveResponse:
    app = request.app

    try:
        session = app.state.sessions.get(body.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    # Cancel any running propagation for this session
    if session.propagation_running:
        session.request_cancel()

    if body.text_prompt is None and body.bbox is None and (not body.points):
        raise HTTPException(status_code=400, detail="At least one of text_prompt, bbox, or points is required")

    t0 = time.perf_counter()

    try:
        result = await app.state.gpu_worker.submit(
            lambda: _run_interactive(
                processor=app.state.image_processor,
                session=session,
                body=body,
            ),
            priority=0,
        )
    except Exception as exc:
        logger.exception("Interactive inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    masks_np: np.ndarray = result["masks"]      # (N, H, W) bool
    scores: list[float] = result["scores"]

    # Write EXR
    exr_path = session.output_dir / f"mask_{body.frame_index:04d}.exr"
    write_mask_exr(exr_path, masks_np)

    return InteractiveResponse(
        session_id=body.session_id,
        frame_index=body.frame_index,
        mask_count=len(masks_np),
        masks_png_b64=masks_to_png_b64_list(masks_np),
        exr_path=str(exr_path),
        scores=scores,
        inference_time_ms=round(elapsed_ms, 1),
    )


# ------------------------------------------------------------------
# Runs inside GPU thread
# ------------------------------------------------------------------

def _run_interactive(processor, session, body: InteractiveRequest) -> dict:
    from PIL import Image  # type: ignore

    # Load the requested frame
    frame_img = _load_frame(session, body.frame_index)

    # Re-encode only if the frame changed since last call
    if session.image_state is None or session.image_state.get("frame_index") != body.frame_index:
        state = processor.set_image(frame_img)
        state["frame_index"] = body.frame_index
        session.image_state = state
    else:
        state = session.image_state

    # Reset prompts from previous call on this frame
    state = processor.reset_all_prompts(state)

    # Text prompt
    if body.text_prompt:
        state = processor.set_text_prompt(prompt=body.text_prompt, state=state)

    # Bounding box
    if body.bbox:
        bb = body.bbox
        # Sam3Processor expects [cx, cy, w, h] normalized
        state = processor.add_geometric_prompt(
            box=[bb.cx, bb.cy, bb.w, bb.h],
            label=True,
            state=state,
        )

    # Points
    if body.points:
        for pt in body.points:
            state = processor.add_geometric_prompt(
                point=[pt.x, pt.y],
                label=pt.label == 1,
                state=state,
            )

    # Keep updated state (backbone features reusable)
    session.image_state = state

    masks = state.get("masks")
    scores_t = state.get("scores")

    if masks is None or len(masks) == 0:
        # Return empty single-mask (all zeros) so the client still gets a valid response
        h, w = session.height, session.width
        masks_np = np.zeros((1, h, w), dtype=bool)
        scores_list = [0.0]
    else:
        import torch
        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().numpy().astype(bool)
        else:
            masks_np = np.asarray(masks, dtype=bool)

        if isinstance(scores_t, torch.Tensor):
            scores_list = scores_t.cpu().tolist()
        else:
            scores_list = list(scores_t) if scores_t is not None else [0.0] * len(masks_np)

    return {"masks": masks_np, "scores": scores_list}


def _load_frame(session, frame_index: int):
    """Load a single frame as a PIL Image from video_path."""
    from PIL import Image  # type: ignore
    import os

    video_path = session.video_path
    path = Path(video_path)

    if path.is_dir():
        # Directory of images: sorted, pick by index
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        frames = sorted(
            f for f in path.iterdir()
            if f.suffix.lower() in exts
        )
        if frame_index >= len(frames):
            raise ValueError(f"frame_index {frame_index} out of range ({len(frames)} frames)")
        return Image.open(frames[frame_index]).convert("RGB")

    # Video file: use OpenCV
    import cv2  # type: ignore
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)
