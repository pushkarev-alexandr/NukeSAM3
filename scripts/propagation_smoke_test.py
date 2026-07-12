"""Propagation smoke test: create_session -> add_prompt -> propagate."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nuke"))

from sam3_client import SAM3Client  # noqa: E402

SEQ = ROOT / "testdata" / "sequence"
OUT = ROOT / "testdata" / "masks"
PROMPT = "dog"
START_FRAME = 0


def main() -> int:
    c = SAM3Client(timeout=600)

    print("=== create_session ===")
    session = c.create_session(str(SEQ), str(OUT), client_id="prop_smoke")
    sid = session["session_id"]
    print(json.dumps(session, indent=2))

    # Video predictor loads frames asynchronously after session create.
    print("\n=== waiting for video session warmup ===")
    time.sleep(45)

    print(f"\n=== add_prompt (frame {START_FRAME}, {PROMPT!r}) ===")
    prompt_resp = None
    for attempt in range(1, 6):
        try:
            prompt_resp = c.add_prompt(sid, START_FRAME, text=PROMPT)
            break
        except Exception as exc:
            print(f"  attempt {attempt} failed: {exc}")
            if attempt == 5:
                raise
            time.sleep(15)
    print(json.dumps(prompt_resp, indent=2))

    print("\n=== propagate ===")
    final = None
    progress_count = 0
    for event in c.iter_propagate(sid, start_frame_index=START_FRAME, output_dir=str(OUT)):
        evt_type = event.get("type", "")
        if evt_type == "progress":
            progress_count += 1
            if progress_count == 1 or progress_count % 10 == 0:
                fi = event.get("frame_index")
                total = event.get("total_frames")
                pct = event.get("percent")
                print(f"  frame {fi}/{total}  {pct}%")
        elif evt_type in ("complete", "cancelled", "error"):
            final = event
            print(json.dumps(event, indent=2))
            break

    written = sorted(OUT.glob("mask_*.exr"))
    print(f"\n=== EXR files: {len(written)} in {OUT} ===")
    if written:
        print(f"  first: {written[0].name} ({written[0].stat().st_size} bytes)")
        print(f"  last:  {written[-1].name} ({written[-1].stat().st_size} bytes)")

    print("\n=== delete_session ===")
    print(json.dumps(c.delete_session(sid), indent=2))

    if final and final.get("type") == "complete":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
