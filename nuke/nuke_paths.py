"""Filesystem path helpers for Nuke knobs and nodes."""


def nuke_path(path: str) -> str:
    """Normalize filesystem paths for Nuke (forward slashes only)."""
    return path.replace("\\", "/") if path else path


def read_path_knob(node, knob_name: str) -> str:
    return nuke_path(node[knob_name].value().strip())


def set_path_knob(node, knob_name: str, path: str) -> None:
    node[knob_name].setValue(nuke_path(path))


def sync_path_knob(node, knob_name: str) -> None:
    raw = node[knob_name].value().strip()
    normalized = nuke_path(raw)
    if normalized != raw:
        node[knob_name].setValue(normalized)
