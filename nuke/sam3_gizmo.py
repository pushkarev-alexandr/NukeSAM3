"""
SAM3 Nuke Gizmo — Python logic layer.

Handles:
  - knobChanged debounce (300ms) → interactive inference
  - Session lifecycle (create / delete on video_path change)
  - Full propagation with nuke.ProgressTask
  - Read node reload after inference

This module is imported by the gizmo's knobChanged callback and button scripts.
It keeps per-node state in a module-level dict (_node_state) so every gizmo
instance is independent.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import threading
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap: make nuke/ directory importable so sam3_client is found
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sam3_client import SAM3Client  # noqa: E402

# ---------------------------------------------------------------------------
# Per-node state
# ---------------------------------------------------------------------------
_node_state: dict[str, dict] = {}

DEBOUNCE_DELAY = 0.35  # seconds


def _get_state(node) -> dict:
    name = node.name()
    if name not in _node_state:
        _node_state[name] = {
            "timer": None,
            "prop_thread": None,
            "cancel_flag": threading.Event(),
            "client": None,
        }
    return _node_state[name]


def _client(node) -> SAM3Client:
    state = _get_state(node)
    if state["client"] is None:
        url = node["server_url"].value().strip() or "http://localhost:8765"
        state["client"] = SAM3Client(base_url=url)
    return state["client"]


# ---------------------------------------------------------------------------
# Knob-changed entry point  (called from gizmo's knobChanged TCL/Python)
# ---------------------------------------------------------------------------

INTERACTIVE_KNOBS = {"bbox", "text_prompt", "point_pos", "neg_point_pos", "confidence"}
SESSION_KNOBS = {"video_path", "output_dir", "server_url"}


def on_create():
    """Initialize prompt UI when the gizmo is created."""
    try:
        import nuke
        _update_prompt_ui(nuke.thisNode())
    except Exception as exc:
        _log_error(exc)


def knob_changed():
    """Call this from the gizmo's knobChanged callback."""
    try:
        import nuke
        node = nuke.thisNode()
        k = nuke.thisKnob()
        name = k.name()

        if name == "prompt_mode":
            _update_prompt_ui(node)
        elif name in SESSION_KNOBS:
            _handle_session_knob_changed(node)
        elif name in INTERACTIVE_KNOBS:
            _schedule_interactive(node)
    except Exception as exc:
        _log_error(exc)


def _update_prompt_ui(node):
    """Show only the knobs for the selected prompt type."""
    mode = node["prompt_mode"].value()
    show_text = mode == "text"
    show_bbox = mode == "bbox"
    show_points = mode == "points"

    node["text_prompt"].setVisible(show_text)
    node["bbox"].setVisible(show_bbox)
    node["point_pos"].setVisible(show_points)
    node["neg_point_pos"].setVisible(show_points)


def _handle_session_knob_changed(node):
    """Re-create session when video path or server URL changes."""
    state = _get_state(node)
    # Reset client on URL change so it's recreated with new URL
    state["client"] = None
    # Close old session
    _close_session(node)
    # Start new session if video_path is set
    video_path = node["video_path"].value().strip()
    if video_path and os.path.exists(video_path):
        t = threading.Thread(target=_create_session_bg, args=(node,), daemon=True)
        t.start()


def _create_session_bg(node):
    """Background thread: create SAM3 session and update gizmo status."""
    import nuke
    try:
        video_path = node["video_path"].value().strip()
        output_dir = node["output_dir"].value().strip()
        if not output_dir:
            output_dir = str(Path(video_path).parent / "sam3_masks")

        resp = _client(node).create_session(
            video_path=video_path,
            output_dir=output_dir,
            client_id=node.name(),
        )
        session_id = resp["session_id"]
        frame_count = resp.get("frame_count", 0)

        def update():
            node["session_id"].setValue(session_id)
            node["output_dir"].setValue(output_dir)
            node["status"].setValue(
                f"Session ready  |  {frame_count} frames  |  {resp.get('width')}x{resp.get('height')}"
            )
            # Point internal Read node at the output directory
            _update_read_node(node, output_dir)

        nuke.executeInMainThread(update)
    except Exception as exc:
        _set_status_mt(node, f"Session error: {exc}")


# ---------------------------------------------------------------------------
# Interactive inference
# ---------------------------------------------------------------------------

def _schedule_interactive(node):
    state = _get_state(node)
    if state["timer"] is not None:
        state["timer"].cancel()
    state["timer"] = threading.Timer(DEBOUNCE_DELAY, _fire_interactive, args=(node,))
    state["timer"].start()


def _collect_prompt(node) -> tuple[Optional[str], Optional[dict], Optional[list]]:
    """Collect prompt values for the active prompt_mode only."""
    mode = node["prompt_mode"].value()
    text = None
    bbox = None
    points = None

    if mode == "text":
        text = node["text_prompt"].value().strip() or None
    elif mode == "bbox":
        try:
            bv = node["bbox"].value()  # Box3_Knob returns (x, y, r, t) in pixels
            w_img = node.width() or 1
            h_img = node.height() or 1
            x, y, r, t = bv[0], bv[1], bv[2], bv[3]
            if r > x and t > y:
                cx = ((x + r) / 2) / w_img
                cy = ((y + t) / 2) / h_img
                bw = (r - x) / w_img
                bh = (t - y) / h_img
                bbox = {"cx": cx, "cy": cy, "w": bw, "h": bh}
        except Exception:
            pass
    elif mode == "points":
        collected = []
        try:
            px, py = node["point_pos"].value()
            w_img = node.width() or 1
            h_img = node.height() or 1
            if px != 0 or py != 0:
                collected.append({"x": px / w_img, "y": py / h_img, "label": 1})
        except Exception:
            pass
        try:
            npx, npy = node["neg_point_pos"].value()
            w_img = node.width() or 1
            h_img = node.height() or 1
            if npx != 0 or npy != 0:
                collected.append({"x": npx / w_img, "y": npy / h_img, "label": 0})
        except Exception:
            pass
        if collected:
            points = collected

    return text, bbox, points


def _fire_interactive(node):
    """Runs in timer thread after debounce delay."""
    import nuke
    try:
        session_id = node["session_id"].value().strip()
        if not session_id:
            return

        frame_index = int(nuke.frame())
        text, bbox, points = _collect_prompt(node)

        if not text and bbox is None and not points:
            return

        confidence = node["confidence"].value()

        resp = _client(node).interactive(
            session_id=session_id,
            frame_index=frame_index,
            text_prompt=text,
            bbox=bbox,
            points=points or None,
            confidence=confidence,
        )

        exr_path = resp.get("exr_path", "")
        scores = resp.get("scores", [])
        score_str = ", ".join(f"{s:.2f}" for s in scores)

        def update():
            _update_read_node(node, None, single_exr=exr_path, frame=frame_index)
            node["status"].setValue(
                f"Frame {frame_index}  |  {resp.get('mask_count', 0)} masks  "
                f"|  scores: [{score_str}]  |  {resp.get('inference_time_ms', 0):.0f}ms"
            )

        nuke.executeInMainThread(update)

    except Exception as exc:
        _set_status_mt(node, f"Infer error: {exc}")


# ---------------------------------------------------------------------------
# Full propagation
# ---------------------------------------------------------------------------

def render_all(node=None):
    """Called by the 'Render All' button."""
    import nuke
    if node is None:
        node = nuke.thisNode()

    session_id = node["session_id"].value().strip()
    if not session_id:
        nuke.message("No active SAM3 session. Set video_path first.")
        return

    text, bbox, points = _collect_prompt(node)
    if not text and bbox is None and not points:
        mode = node["prompt_mode"].value()
        nuke.message(f"Set a {mode} prompt before rendering.")
        return

    state = _get_state(node)
    if state["prop_thread"] and state["prop_thread"].is_alive():
        nuke.message("Propagation already running.")
        return

    # Create ProgressTask in main thread before spawning worker
    task = nuke.ProgressTask("SAM3 Propagation")
    state["cancel_flag"].clear()

    output_dir = node["output_dir"].value().strip()
    frame_index = int(nuke.frame())

    state["prop_thread"] = threading.Thread(
        target=_propagation_worker,
        args=(node, session_id, output_dir, frame_index, text, bbox, points, task, state["cancel_flag"]),
        daemon=True,
    )
    state["prop_thread"].start()


def cancel_render(node=None):
    """Called by the 'Cancel' button."""
    import nuke
    if node is None:
        node = nuke.thisNode()
    state = _get_state(node)
    state["cancel_flag"].set()


def _propagation_worker(
    node,
    session_id: str,
    output_dir: str,
    frame_index: int,
    text: Optional[str],
    bbox: Optional[dict],
    points: Optional[list],
    task,
    cancel_flag: threading.Event,
):
    import nuke

    def set_progress(pct: int, msg: str = ""):
        def _do():
            task.setProgress(pct)
            if msg:
                task.setMessage(msg)
        nuke.executeInMainThread(_do)

    try:
        set_progress(0, "Adding prompt…")
        client = _client(node)
        if text:
            client.add_prompt(session_id, frame_index, text=text)
        elif bbox:
            client.add_prompt(
                session_id,
                frame_index,
                bounding_boxes=[[bbox["cx"], bbox["cy"], bbox["w"], bbox["h"]]],
                bounding_box_labels=[1],
            )
        elif points:
            client.add_prompt(
                session_id,
                frame_index,
                points=[[p["x"], p["y"]] for p in points],
                point_labels=[p["label"] for p in points],
            )

        set_progress(0, "Starting propagation…")

        for event in client.iter_propagate(
            session_id=session_id,
            output_dir=output_dir if output_dir else None,
            cancel_check=lambda: cancel_flag.is_set(),
        ):
            evt_type = event.get("type", "")
            if evt_type == "progress":
                pct = event.get("percent", 0)
                fi = event.get("frame_index", 0)
                total = event.get("total_frames", 1)
                elapsed = event.get("elapsed_ms", 0) / 1000
                set_progress(pct, f"Frame {fi}/{total}  {elapsed:.1f}s")

            elif evt_type == "complete":
                total_written = event.get("total_frames_written", 0)
                out = event.get("output_dir", output_dir)

                def on_complete(out=out, total_written=total_written):
                    _update_read_node(node, out)
                    node["status"].setValue(
                        f"Done  |  {total_written} frames  |  {out}"
                    )
                    del task  # closes progress bar

                nuke.executeInMainThread(on_complete)
                return

            elif evt_type in ("cancelled", "error"):
                msg = event.get("message", event.get("reason", "unknown"))
                def on_cancel(msg=msg):
                    node["status"].setValue(f"Cancelled: {msg}")
                    del task
                nuke.executeInMainThread(on_cancel)
                return

    except Exception as exc:
        def on_err(exc=exc):
            node["status"].setValue(f"Propagation error: {exc}")
            del task
        nuke.executeInMainThread(on_err)


# ---------------------------------------------------------------------------
# Read node management
# ---------------------------------------------------------------------------

def _update_read_node(node, output_dir: Optional[str], single_exr: Optional[str] = None, frame: Optional[int] = None):
    """Update the internal Read node to point to mask output."""
    import nuke
    read = node.node("SAM3_masks_read") if hasattr(node, "node") else None
    if read is None:
        return

    if single_exr:
        read["file"].setValue(single_exr)
        if frame is not None:
            read["first"].setValue(frame)
            read["last"].setValue(frame)
    elif output_dir:
        pattern = str(Path(output_dir) / "mask_%04d.exr")
        read["file"].setValue(pattern)
        # Let Nuke detect the frame range
        read.knob("reload").execute()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _close_session(node):
    session_id = node["session_id"].value().strip()
    if session_id:
        try:
            _client(node).delete_session(session_id)
        except Exception:
            pass
        node["session_id"].setValue("")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _set_status_mt(node, msg: str):
    try:
        import nuke
        nuke.executeInMainThread(lambda: node["status"].setValue(msg))
    except Exception:
        pass


def _log_error(exc: Exception):
    try:
        import nuke
        nuke.warning(f"SAM3 gizmo error: {exc}")
    except Exception:
        print(f"SAM3 gizmo error: {exc}")

if __name__ == "__main__":
    knob_changed()
