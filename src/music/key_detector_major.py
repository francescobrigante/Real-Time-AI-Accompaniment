# =============================================================================
# Minimal Krumhansl-Schmuckler Key Detector (Major Keys Only)
# =============================================================================

import numpy as np
import mido
from typing import List, Tuple, Optional
from collections import deque
from src.config import MAJOR_PROFILE, NOTES, INPUT_PORT

class KeyDetector:
    
    def __init__(self, min_notes: int = 4, min_confidence: float = 0.6):
        self.min_notes = min_notes
        self.min_confidence = min_confidence
        
        # Pre-compute rotated profiles for all 12 keys
        self.profiles = np.zeros((12, 12), dtype=np.float32)
        for i in range(12):
            self.profiles[i] = np.roll(MAJOR_PROFILE, i)
            
        # Characteristic intervals for Major keys (Major 3rd, Major 7th)
        self.unique_intervals = {i: {4, 11} for i in range(12)}

    def detect(self, notes: List[int]) -> Optional[Tuple[str, float]]:
        """
        Detects major key from a list of MIDI notes.
        Returns: (Key Name, Confidence) or None
        """
        if len(notes) < self.min_notes:
            return None
            
        # 1. Create Pitch Class Histogram
        hist = np.zeros(12, dtype=np.float32)
        for n in notes:
            hist[n % 12] += 1
        
        # Normalize
        if hist.sum() > 0:
            hist /= hist.sum()
        else:
            return None
            
        # 2. Compute Correlation with all 12 Major Profiles
        # Pearson correlation: cov(X,Y) / (std(X) * std(Y))
        
        # Center data
        h_mean = hist.mean()
        p_mean = self.profiles.mean(axis=1, keepdims=True)
        
        h_centered = hist - h_mean
        p_centered = self.profiles - p_mean
        
        # Covariance
        cov = (p_centered * h_centered).mean(axis=1)
        
        # Standard Deviations
        h_std = hist.std()
        p_std = self.profiles.std(axis=1)
        
        # Correlation
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = cov / (h_std * p_std)
            corr = np.nan_to_num(corr)
            
        # 3. Find Best Match
        best_idx = np.argmax(corr)
        best_corr = corr[best_idx]
        
        # Normalize correlation to 0-1 confidence
        confidence = max(0.0, min(1.0, (best_corr + 1) / 2))
        
        # 4. Validate with Characteristic Notes (Unique Intervals)
        # Check if we have at least one characteristic note (Major 3rd or Major 7th)
        pitch_classes = set(n % 12 for n in notes)
        relative_pcs = {(pc - best_idx) % 12 for pc in pitch_classes}
        
        has_unique = bool(self.unique_intervals[best_idx] & relative_pcs)
        
        if not has_unique:
            confidence *= 0.5 # Penalty if no characteristic note found
            
        if confidence < self.min_confidence:
            return None
            
        return NOTES[best_idx], confidence

# ======================================= Real-Time Test (VMPK Input) ==========================================

if __name__ == "__main__":
    WINDOW_SIZE = 12
    
    print(f"🎹 Listening on '{INPUT_PORT}'...")
    print(f"   (Press Ctrl+C to stop)")
    print("-" * 40)
    
    detector = KeyDetector()
    note_window = deque(maxlen=WINDOW_SIZE)
    
    try:
        with mido.open_input(INPUT_PORT) as inport:
            for msg in inport:
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Add note to window
                    note_window.append(msg.note)
                    
                    # Detect Key
                    result = detector.detect(list(note_window))
                    
                    # Visualization
                    notes_str = " ".join([NOTES[n % 12] for n in note_window])
                    
                    if result:
                        key, conf = result
                        bar = "█" * int(conf * 10)
                        print(f"\rNotes: [{notes_str:<30}]  ->  Key: {key} Major  ({conf:.2f}) {bar}", end="")
                    else:
                        print(f"\rNotes: [{notes_str:<30}]  ->  ... analyzing ...", end="")
                        
    except OSError:
        print(f"\n[ERROR] Could not open MIDI port '{INPUT_PORT}'.")
        print("Available ports:", mido.get_input_names())
    except KeyboardInterrupt:
        print("\n\nStopped.")