# Core refactored components
try:
    from .track import Track
    from .segment import Segment
    from .mashup_engine import MashupEngine
    __all__ = ['Track', 'Segment', 'MashupEngine']
except ImportError:
    # Fallback for backward compatibility
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from automashup.src.track import Track
    from automashup.src.segment import Segment
    __all__ = ['Track', 'Segment']

