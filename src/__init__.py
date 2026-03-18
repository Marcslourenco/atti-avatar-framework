"""ATTI Avatar Framework"""

from .tts.xtts_engine import XTTSEngine
from .avatar.liveportrait_engine import LivePortraitEngine
from .media.pipeline import MediaPipeline
from .multimodal.atti_adapter import ATTIMultimodalAdapter

__version__ = "1.0.0"
__all__ = [
    "XTTSEngine",
    "LivePortraitEngine",
    "MediaPipeline",
    "ATTIMultimodalAdapter",
]
