"""
SAM3 server HTTP client — zero external dependencies (stdlib only).
Works inside Nuke's embedded Python environment.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Generator, Optional


class SAM3Client:
    def __init__(self, base_url: str = "http://localhost:8765", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, body: dict, timeout: Optional[int] = None) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout if timeout is not None else self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {body_text}") from e

    def _delete(self, path: str) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}") from e

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}") from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> dict:
        return self._get("/health")

    def create_session(self, video_path: str, output_dir: str, client_id: str = "") -> dict:
        return self._post("/session/create", {
            "video_path": video_path,
            "output_dir": output_dir,
            "client_id": client_id,
        })

    def delete_session(self, session_id: str) -> dict:
        return self._delete(f"/session/{session_id}")

    def add_prompt(
        self,
        session_id: str,
        frame_index: int,
        text: Optional[str] = None,
        bounding_boxes: Optional[list] = None,
        bounding_box_labels: Optional[list] = None,
        points: Optional[list] = None,
        point_labels: Optional[list] = None,
    ) -> dict:
        body: dict = {
            "session_id": session_id,
            "frame_index": frame_index,
        }
        if text is not None:
            body["text"] = text
        if bounding_boxes is not None:
            body["bounding_boxes"] = bounding_boxes
            body["bounding_box_labels"] = bounding_box_labels or [1] * len(bounding_boxes)
        if points is not None:
            body["points"] = points
            body["point_labels"] = point_labels or [1] * len(points)
        return self._post("/infer/add_prompt", body, timeout=60)

    def interactive(
        self,
        session_id: str,
        frame_index: int,
        text_prompt: Optional[str] = None,
        bbox: Optional[dict] = None,
        points: Optional[list] = None,
        confidence: float = 0.5,
    ) -> dict:
        body: dict = {
            "session_id": session_id,
            "frame_index": frame_index,
            "confidence_threshold": confidence,
        }
        if text_prompt:
            body["text_prompt"] = text_prompt
        if bbox:
            body["bbox"] = bbox
        if points:
            body["points"] = points
        return self._post("/infer/interactive", body)

    def reset_session(self, session_id: str) -> dict:
        return self._post(f"/session/{session_id}/reset", {})

    def iter_propagate(
        self,
        session_id: str,
        start_frame_index: int = 0,
        direction: str = "both",
        output_dir: Optional[str] = None,
        filename_pattern: str = "mask_%04d.exr",
        cancel_check=None,
    ) -> Generator[dict, None, None]:
        """
        Generator that yields SSE event dicts:
          {"type": "progress"|"complete"|"cancelled"|"error", ...}

        cancel_check: optional callable() → bool, return True to stop iteration.
        """
        body: dict = {
            "session_id": session_id,
            "start_frame_index": start_frame_index,
            "propagation_direction": direction,
            "frame_filename_pattern": filename_pattern,
        }
        if output_dir:
            body["output_dir"] = output_dir

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/infer/propagate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=3600) as resp:
            event_type = "message"
            for raw_line in resp:
                if cancel_check and cancel_check():
                    break
                line = raw_line.decode("utf-8").rstrip("\n\r")
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
                    payload["type"] = event_type
                    yield payload
                    event_type = "message"

if __name__ == "__main__":
    client = SAM3Client()
    print(client.health())
