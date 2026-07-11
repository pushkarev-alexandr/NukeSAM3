import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


class Config:
    # Session settings
    SESSION_TTL_SECONDS: int = int(os.getenv("SAM3_SESSION_TTL", "1200"))  # 20 min
    SESSION_WATCHDOG_INTERVAL: int = 60  # seconds between cleanup passes

    # GPU worker
    QUEUE_MAXSIZE: int = 256

    # Mask output
    DEFAULT_OUTPUT_DIR: Path = Path(
        os.getenv("SAM3_OUTPUT_DIR", str(_REPO_ROOT / "testdata" / "masks"))
    )
    EXR_FILENAME_PATTERN: str = "mask_%04d.exr"


config = Config()
