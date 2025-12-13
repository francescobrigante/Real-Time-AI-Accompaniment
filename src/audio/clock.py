# ==============================================================================
# Conductor - Centralized Clock for Real-Time Synchronization
# ==============================================================================
# The Conductor is the single source of truth for timing in the pipeline.
# All modules (metronome, playback, prediction) query the Conductor for timing.
# ==============================================================================

import time
import threading
from typing import Callable, Optional

from src.config import DEFAULT_BPM, DEFAULT_BEATS_PER_BAR, BPM_MIN, BPM_MAX, DELTA_RETAIN_BPM
from src.utils.logger import setup_logger

logger = setup_logger()


class Conductor:
    """
    Provides synchronized timing for all modules by maintaining:
    - Current BPM (with thread-safe access)
    - Absolute time calculations for bars
    """
    
    def __init__(self, 
                 initial_bpm: float = DEFAULT_BPM,
                 beats_per_bar: float = DEFAULT_BEATS_PER_BAR,
                 on_tempo_change: Optional[Callable[[float, float], None]] = None):

        # Core timing state
        self._bpm = self._clamp_bpm(initial_bpm)
        self._beats_per_bar = beats_per_bar
        self._start_time: Optional[float] = None        # time when the conductor started
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks
        self._on_tempo_change = on_tempo_change         # callback for when the tempo changes

    # --------------------------------------------------------------------------
    # Properties (thread-safe)
    # --------------------------------------------------------------------------
    
    @property
    def bpm(self) -> float:
        with self._lock:
            return self._bpm
    
    @property
    def beat_duration(self) -> float:
        with self._lock:
            return 60.0 / self._bpm
    
    @property
    def bar_duration(self) -> float:
        with self._lock:
            return self._beats_per_bar * (60.0 / self._bpm)
    
    @property
    def is_running(self) -> bool:
        return self._start_time is not None
    
    @property
    def start_time(self) -> Optional[float]:
        with self._lock:
            return self._start_time
            
    # --------------------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------------------
    
    def start(self) -> float:
        with self._lock:
            self._start_time = time.time()
            # logger.info(f"[CONDUCTOR] Started at {self._bpm:.1f} BPM")
            return self._start_time
    
    def reset(self):
        with self._lock:
            self._start_time = None

    def stop(self):
        self.reset()
        logger.info("[CONDUCTOR] Stopped")
    
    # force the tempo to a specific value
    def force_tempo(self, new_bpm: float):
        new_bpm = self._clamp_bpm(new_bpm)
        
        with self._lock:
            old_bpm = self._bpm
            self._bpm = new_bpm
            
            # notify the callback if the tempo has changed by more than DELTA_RETAIN_BPM
            if self._on_tempo_change and abs(old_bpm - self._bpm) > DELTA_RETAIN_BPM:
                self._on_tempo_change(old_bpm, self._bpm)
    
    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    
    @staticmethod
    def _clamp_bpm(bpm: float) -> float:
        return max(BPM_MIN, min(BPM_MAX, bpm))
    
    # Representation for debugging
    def __repr__(self) -> str:
        state = "running" if self.is_running else "stopped"
        return f"Conductor({self._bpm:.1f} BPM, {state})"
