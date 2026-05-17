# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A SAM3 inference server (`server/`) and Nuke gizmo (`nuke/`) that let Nuke artists on a LAN send segmentation requests to a single GPU machine. `sam3/` is a git submodule (Meta's SAM3 fork) — treat it as read-only.

## Running the server

All commands from the repo root, using the venv in `sam3/.venv`:

```bash
# Activate venv
sam3\.venv\Scripts\activate

# Start the server (loads both SAM3 models, then listens on :8765)
python -m server.main

# Override defaults via env vars
SAM3_PORT=9000 SAM3_VERSION=sam3 python -m server.main
```

Swagger UI at `http://localhost:8765/docs` once running.

## Testing

```bash
# SAM3 submodule tests (run from sam3/)
cd sam3
pytest test/

# Quick server health check (no Nuke needed)
python nuke/sam3_client.py

# Test propagation SSE stream end-to-end
python -c "
from nuke.sam3_client import SAM3Client
c = SAM3Client()
sid = c.create_session('path/to/frames', 'C:/tmp/masks')['session_id']
c.add_prompt(sid, 0, text='person')
for e in c.iter_propagate(sid): print(e)
"
```

## Architecture

### Data flow

```
Nuke gizmo (any LAN machine)
  └── nuke/sam3_client.py  (urllib, no deps)
        │ HTTP/SSE
        ▼
server/main.py  (FastAPI, port 8765)
  ├── gpu_worker.py  ← single thread, PriorityQueue
  │     priority 0 = interactive (one frame, ~200ms)
  │     priority 1 = propagation (all frames, streaming)
  ├── session_manager.py  ← per-client session registry + TTL watchdog
  └── routes/
        interactive.py  → Sam3Processor (image model, fast)
        propagate.py    → build_sam3_predictor (video model, temporal consistency)
        session.py      → create/delete sessions
        health.py       → GPU stats
```

### Two SAM3 models loaded at startup

- **`Sam3Processor`** (image model) — used for interactive per-frame inference. Stateless per call; caches backbone features in `session.image_state` per session.
- **`build_sam3_predictor`** (video predictor) — used for full-video propagation. Maintains its own internal session registry; our `sam3_video_session_id` maps into it.

### Multi-user concurrency

Single GPU → all inference is serialised through `gpu_worker.py`. Interactive requests preempt propagation **on the same session**: the `session.propagation_cancel` threading.Event triggers SAM3's own `cancel_propagation` request. Propagation from a different session waits in the queue.

### Propagation streaming

`POST /infer/propagate` returns `text/event-stream`. Each frame yields an SSE `progress` event with `{frame_index, percent, exr_path}`. Final event is `complete` or `cancelled`. The Nuke gizmo reads this line-by-line in a background thread and calls `nuke.executeInMainThread` to update `nuke.ProgressTask`.

### Mask format

Multi-channel half-float EXR per frame, channels named `mask_0`, `mask_1`, … (one per detected object). Interactive response also includes `masks_png_b64` (base64 PNG list) for immediate viewer feedback without a file round-trip. Falls back to PNG if `OpenEXR` is not installed.

### Nuke gizmo wiring

`nuke/sam3_gizmo.py` is imported by the gizmo's `knobChanged` callback. Per-node state (debounce timer, propagation thread, cancel flag) lives in the module-level `_node_state` dict keyed by node name. The gizmo contains an internal `Read` node (`SAM3_masks_read`) that is redirected to the EXR output path after each inference.

To install: add to `~/.nuke/menu.py`:
```python
import sys; sys.path.insert(0, r"C:/path/to/NukeSAM3/nuke")
import menu
```

## Key env vars

| Variable | Default | Purpose |
|---|---|---|
| `SAM3_VERSION` | `sam3` | `sam3` or `sam3.1` |
| `SAM3_DEVICE` | `cuda` | PyTorch device |
| `SAM3_CHECKPOINT` | `None` | Local checkpoint path; auto-downloads from HF if unset |
| `SAM3_PORT` | `8765` | Server listen port |
| `SAM3_SESSION_TTL` | `1200` | Seconds before idle session is closed |
| `SAM3_OUTPUT_DIR` | `C:/tmp/sam3_masks` | Default EXR output directory |

## Dependencies

Server dependencies are in `server/requirements.txt`. Install into the sam3 venv:
```bash
pip install -r server/requirements.txt
```

Key additions on top of SAM3's own deps: `fastapi`, `uvicorn[standard]`, `sse-starlette`, `Pillow`, `OpenEXR`, `opencv-python`.
