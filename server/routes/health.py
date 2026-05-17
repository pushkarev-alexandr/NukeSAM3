from __future__ import annotations

from fastapi import APIRouter, Request

from ..models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    app = request.app
    gpu_name = "N/A"
    gpu_free_gb = 0.0
    gpu_total_gb = 0.0

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            gpu_free_gb = free_bytes / 1e9
            gpu_total_gb = total_bytes / 1e9
    except Exception:
        pass

    return HealthResponse(
        status="ready" if app.state.model_loaded else "loading",
        gpu_name=gpu_name,
        gpu_memory_free_gb=round(gpu_free_gb, 2),
        gpu_memory_total_gb=round(gpu_total_gb, 2),
        active_sessions=app.state.sessions.active_count(),
        queue_depth=app.state.gpu_worker.queue_depth,
        model_loaded=app.state.model_loaded,
    )
