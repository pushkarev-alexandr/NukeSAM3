"""
Client session registry.

Each Nuke gizmo instance gets one NukeClientSession.
The session holds:
  - image_state: Sam3Processor state (cached backbone features for last interactive frame)
  - sam3_video_session_id: ID inside the video predictor's internal registry
  - propagation_cancel: threading.Event to interrupt a running propagation
  - last_active: timestamp for TTL-based expiry
"""
from __future__ import annotations

import asyncio
import logging
import time
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class NukeClientSession:
    session_id: str
    video_path: str
    output_dir: Path
    frame_count: int
    width: int
    height: int
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # SAM3 internal state
    sam3_video_session_id: Optional[str] = None
    image_state: Optional[dict] = None       # Sam3Processor state dict

    # Propagation control
    propagation_cancel: threading.Event = field(default_factory=threading.Event)
    propagation_running: bool = False

    def touch(self) -> None:
        self.last_active = time.time()

    def is_expired(self, ttl: int) -> bool:
        return (time.time() - self.last_active) > ttl

    def request_cancel(self) -> None:
        self.propagation_cancel.set()

    def reset_cancel(self) -> None:
        self.propagation_cancel.clear()


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, NukeClientSession] = {}
        self._lock = threading.Lock()
        self._watchdog_task: Optional[asyncio.Task] = None

        # Set by main.py after model loading
        self.video_predictor: Any = None
        self.gpu_worker: Any = None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        video_path: str,
        output_dir: Optional[str],
        client_id: Optional[str],
        frame_count: int,
        width: int,
        height: int,
        sam3_video_session_id: str,
    ) -> NukeClientSession:
        session = NukeClientSession(
            session_id=str(uuid.uuid4()),
            video_path=video_path,
            output_dir=Path(output_dir) if output_dir else config.DEFAULT_OUTPUT_DIR,
            frame_count=frame_count,
            width=width,
            height=height,
            sam3_video_session_id=sam3_video_session_id,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        logger.info("Session created: %s  video=%s  client=%s", session.session_id, video_path, client_id)
        return session

    def get(self, session_id: str) -> NukeClientSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        session.touch()
        return session

    def delete(self, session_id: str) -> Optional[NukeClientSession]:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            logger.info("Session deleted: %s", session_id)
        return session

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------
    # Expiry watchdog
    # ------------------------------------------------------------------

    async def start_watchdog(self) -> None:
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def stop_watchdog(self) -> None:
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(config.SESSION_WATCHDOG_INTERVAL)
            await self._cleanup_expired()

    async def _cleanup_expired(self) -> None:
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if s.is_expired(config.SESSION_TTL_SECONDS)
            ]

        for sid in expired:
            logger.info("Session expired, closing: %s", sid)
            await self._close_session_async(sid)

    async def _close_session_async(self, session_id: str) -> None:
        session = self.delete(session_id)
        if session is None:
            return
        if self.video_predictor and session.sam3_video_session_id:
            loop = asyncio.get_event_loop()
            predictor = self.video_predictor
            vid_sid = session.sam3_video_session_id
            await loop.run_in_executor(
                None,
                lambda: predictor.handle_request(
                    {"type": "close_session", "session_id": vid_sid}
                ),
            )
