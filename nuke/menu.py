"""
Nuke menu registration for SAM3 gizmo.

Add to your ~/.nuke/menu.py or project's init.py:
    import sys, os
    sys.path.insert(0, r"C:/path/to/NukeSAM3/nuke")
    import menu  # noqa: F401
"""
# import os
# import sys
import nuke

# _NUKE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Make the nuke/ directory importable inside Nuke
# if _NUKE_DIR not in sys.path:
#     sys.path.insert(0, _NUKE_DIR)

# # Register the gizmo path so nuke.createNode("SAM3") works
# nuke.pluginAddPath(_NUKE_DIR)


def _check_server_health():
    import nuke
    from sam3_client import SAM3Client
    url = nuke.getInput("SAM3 Server URL", "http://localhost:8765")
    if not url:
        return
    try:
        client = SAM3Client(base_url=url)
        info = client.health()
        msg = (
            f"Status:    {info.get('status')}\n"
            f"GPU:       {info.get('gpu_name')}\n"
            f"VRAM free: {info.get('gpu_memory_free_gb'):.1f} GB / {info.get('gpu_memory_total_gb'):.1f} GB\n"
            f"Sessions:  {info.get('active_sessions')}\n"
            f"Queue:     {info.get('queue_depth')}"
        )
        nuke.message(msg)
    except Exception as exc:
        nuke.message(f"Cannot reach server:\n{exc}")


# Add to toolbar
toolbar = nuke.toolbar("Nodes")
sam3_menu = toolbar.addMenu("SAM3", icon="")

sam3_menu.addCommand("SAM3 Segmentation", "nuke.createNode('SAM3')")
sam3_menu.addCommand("Check Server Health", _check_server_health)
