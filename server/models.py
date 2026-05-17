from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


# ---------- Session ----------

class SessionCreateRequest(BaseModel):
    video_path: str
    output_dir: Optional[str] = None
    client_id: Optional[str] = None


class SessionCreateResponse(BaseModel):
    session_id: str
    frame_count: int
    width: int
    height: int
    expires_in_seconds: int


class SessionDeleteResponse(BaseModel):
    is_success: bool
    gpu_memory_freed_gb: float = 0.0


# ---------- Prompts ----------

class BBoxPrompt(BaseModel):
    cx: float = Field(..., ge=0.0, le=1.0, description="Center X, normalized")
    cy: float = Field(..., ge=0.0, le=1.0, description="Center Y, normalized")
    w: float = Field(..., ge=0.0, le=1.0, description="Width, normalized")
    h: float = Field(..., ge=0.0, le=1.0, description="Height, normalized")


class PointPrompt(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    label: int = Field(1, description="1=positive, 0=negative")


# ---------- Interactive inference ----------

class InteractiveRequest(BaseModel):
    session_id: str
    frame_index: int
    text_prompt: Optional[str] = None
    bbox: Optional[BBoxPrompt] = None
    points: Optional[list[PointPrompt]] = None
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)


class InteractiveResponse(BaseModel):
    session_id: str
    frame_index: int
    mask_count: int
    masks_png_b64: list[str]
    exr_path: str
    scores: list[float]
    inference_time_ms: float


# ---------- Propagation ----------

class PropagateRequest(BaseModel):
    session_id: str
    start_frame_index: int = 0
    propagation_direction: str = Field("both", pattern="^(forward|backward|both)$")
    output_dir: Optional[str] = None
    frame_filename_pattern: str = "mask_%04d.exr"


# SSE events are plain dicts serialized to JSON strings

# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str
    gpu_name: str
    gpu_memory_free_gb: float
    gpu_memory_total_gb: float
    active_sessions: int
    queue_depth: int
    model_loaded: bool
