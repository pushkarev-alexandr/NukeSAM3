"""
SAM3 Inference Server

Start with:
    cd NukeSAM3
    uvicorn server.main:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

# Make sam3 importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "sam3"))

from .config import config
from .gpu_worker import GPUWorker
from .session_manager import SessionManager
from .routes.health import router as health_router
from .routes.session import router as session_router
from .routes.interactive import router as interactive_router
from .routes.propagate import router as propagate_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading SAM3 models …")
    app.state.model_loaded = False

    from sam3 import build_sam3_image_model, build_sam3_predictor
    from sam3.model.sam3_image_processor import Sam3Processor

    # Image model for interactive single-frame inference
    image_model = build_sam3_image_model()
    image_processor = Sam3Processor(image_model)

    # Video predictor for full-video propagation
    video_predictor = build_sam3_predictor(
        version="sam3",
        apply_temporal_disambiguation=False,
    )

    gpu_worker = GPUWorker()
    gpu_worker.start()

    sessions = SessionManager()
    sessions.video_predictor = video_predictor
    sessions.gpu_worker = gpu_worker

    app.state.image_processor = image_processor
    app.state.video_predictor = video_predictor
    app.state.gpu_worker = gpu_worker
    app.state.sessions = sessions
    app.state.model_loaded = True

    logger.info("SAM3 models ready.")

    await sessions.start_watchdog()

    yield

    logger.info("Shutting down …")
    await sessions.stop_watchdog()
    gpu_worker.stop()


app = FastAPI(title="SAM3 Inference Server", version="1.0.0", lifespan=lifespan)

app.include_router(health_router)
app.include_router(session_router)
app.include_router(interactive_router)
app.include_router(propagate_router)
