import os
from pathlib import Path


class Config:
    HOST: str = os.getenv("SAM3_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("SAM3_PORT", "8765"))

    # SAM3 model settings
    SAM3_VERSION: str = os.getenv("SAM3_VERSION", "sam3")
    SAM3_DEVICE: str = os.getenv("SAM3_DEVICE", "cuda")
    SAM3_CHECKPOINT: str | None = os.getenv("SAM3_CHECKPOINT", None)
    BPE_PATH: str | None = os.getenv("SAM3_BPE_PATH", None)

    # Session settings
    SESSION_TTL_SECONDS: int = int(os.getenv("SAM3_SESSION_TTL", "1200"))  # 20 min
    SESSION_WATCHDOG_INTERVAL: int = 60  # seconds between cleanup passes

    # GPU worker
    QUEUE_MAXSIZE: int = 256

    # Mask output
    DEFAULT_OUTPUT_DIR: Path = Path(os.getenv("SAM3_OUTPUT_DIR", "C:/tmp/sam3_masks"))
    EXR_FILENAME_PATTERN: str = "mask_%04d.exr"

    # Interactive inference
    DEFAULT_CONFIDENCE: float = 0.5
    IMAGE_RESOLUTION: int = 1008


config = Config()
