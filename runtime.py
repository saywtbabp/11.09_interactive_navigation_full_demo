"""兼容 PyCharm 直接启动 Conda 解释器时未激活的 Windows DLL 路径。"""
import os
import sys
from pathlib import Path

_dll_handles = []
# Runtime-only ONNX wheels are isolated from the user's Conda installation.
_vendor = Path(__file__).resolve().parent / '.vendor'
if _vendor.is_dir() and str(_vendor) not in sys.path:
    sys.path.insert(0, str(_vendor))
if os.name == "nt":
    dll_dir = Path(sys.prefix) / "Library" / "bin"
    if dll_dir.is_dir():
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _dll_handles.append(os.add_dll_directory(str(dll_dir)))
