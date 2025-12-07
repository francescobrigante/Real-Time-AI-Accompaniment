# ===============================================================================
#                                      Predictor
# ===============================================================================
# Predicts next chord given a history of chords and a history of played notes
# Dual pipeline:
#   - LSTM: Predicts next chord given a history of chords
#   - Ear: Predicts next chord given a history of played notes
#
# Predicted probability distributions are merged to select the final prediction
# ===============================================================================

from typing import Optional
from collections import deque
import concurrent.futures
import threading

from src.model.ai_harmony import AIHarmonyRules
from src.music.ear import Ear
from src.music.chord import Chord
from src.music.key_detector_major import KeyDetector
from src.utils.music_theory import roman_to_chord, roman_to_compact, get_top_k, format_distribution
from src.utils.logger import setup_logger
from src.config import CHORDS_TO_PRECOMPUTE

logger = setup_logger()

class Predictor:
    def __init__(self, key: str, bpm: int, beats_per_bar: float, window_size: int):
        self.key = key
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.window_size = window_size
        
        # Models
        self.ai = AIHarmonyRules(key)
        self.ear = Ear(key)
        self.key_detector = KeyDetector(min_notes=3)
        
        # State
        self.chord_window = deque(maxlen=window_size)           # list of the last window_size generated chords
        self.precomputed_sequence = []                          # list of precomputed chords (for the future)
        self.precomputed_idx = 0                                # index of the current precomputed chord
        
        # Concurrency
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.lock = threading.Lock()

    # change key   
    def set_key(self, key: str):
        self.key = key
        self.ai.set_key(key)
        self.ear.set_key(key)

    # Updates chord history given a confirmed chord for next prediction
    # Non-blocking to avoid stalling the timing thread   
    def update_history(self, roman_chord: str):

        # 1. Update window (Thread-safe)
        with self.lock:
            # Check match
            prediction_match = False
            expected_roman = None
            if self.precomputed_sequence and self.precomputed_idx < len(self.precomputed_sequence):
                expected_roman, _ = self.precomputed_sequence[self.precomputed_idx]
                if roman_chord == expected_roman:
                    prediction_match = True

            # Update history
            self.chord_window.append(roman_chord)
            current_window = list(self.chord_window)

            should_recompute = False
            # Match
            if prediction_match:
                chord_name = roman_to_compact(self.key, roman_chord)
                logger.info(f"[LSTM] Prediction MATCH: {chord_name} ({roman_chord}).")
                self.precomputed_idx += 1
                
                # Check refill
                remaining = len(self.precomputed_sequence) - self.precomputed_idx
                if remaining < 2:
                    logger.info("[LSTM] Buffer running low, refill...")
                    should_recompute = True
            # Mismatch
            else:
                expected_name = roman_to_compact(self.key, expected_roman) if expected_roman else "None"
                got_name = roman_to_compact(self.key, roman_chord)
                logger.info(f"[LSTM] Prediction MISMATCH: LSTM: {expected_name} ({expected_roman}), Ear: {got_name} ({roman_chord}). Recomputing...")
                should_recompute = True
                
        # 3. Fire-and-Forget Computation
        if should_recompute:
            # Cancel previous if pending
            self.future = self.executor.submit(self._run_precompute, current_window)

    # single worker task for async LSTM inference precomputing chords
    def _run_precompute(self, window_copy):
        try:
            # heavy blocking call: LSTM precomputation of chords
            new_sequence = self.ai.precompute_sequence(window_copy, CHORDS_TO_PRECOMPUTE)
            
            with self.lock:
                self.precomputed_sequence = new_sequence
                self.precomputed_idx = 0
                logger.debug("[LSTM] Async computation complete.")
                
        except Exception as e:
            logger.error(f"[LSTM] Async computation failed: {e}")

    # Returns the next predicted chord from the precomputed sequence 
    # Blocks timing thread if computation is still running (rare)  
    def get_next_prediction(self) -> Optional[Chord]:

        # Wait if future is running (Sync point)
        if self.future and not self.future.done():
            # logger.debug("[LSTM] Waiting for async prediction...")
            self.future.result() # Blocks until done
            
        with self.lock:
            # Lazy Init if empty
            if not self.precomputed_sequence:
                 if self.chord_window:
                     logger.info("[LSTM] Buffer empty, running computation...")
                     self.precomputed_sequence = self.ai.precompute_sequence(list(self.chord_window), CHORDS_TO_PRECOMPUTE)
                     self.precomputed_idx = 0
                 else:
                     return None
                     
            if self.precomputed_idx < len(self.precomputed_sequence):
                roman, _ = self.precomputed_sequence[self.precomputed_idx]
                root, chord_type = roman_to_chord(self.key, roman)
                return Chord(root, chord_type, self.bpm, self.beats_per_bar)
            else:
                logger.warning("[LSTM] Buffer exhausted!")
                return None

    # Returns the distribution for the current prediction step   
    def get_current_distribution(self) -> dict:
        with self.lock:
            if self.precomputed_sequence and self.precomputed_idx < len(self.precomputed_sequence):
                _, dist = self.precomputed_sequence[self.precomputed_idx]
                return dist
            return {}

    # MAIN PREDICTION LOGIC: Refines the prediction based on played notes (Ear)
    # combines LSTM and Ear predictions by multiplying their distributions
    def refine_prediction(self, scheduled_chord: Chord, midi_listener) -> Chord:
        if not midi_listener:
            return scheduled_chord
        
        # 1. Get played notes by soloist
        note_window = midi_listener.get_note_window()
        if not note_window:
            return None # no notes played, return None which will be handled as WAIT FOR INPUT
            
        # 2. Extract MIDI notes
        midi_notes = [note[0] for note in note_window]
        
        # 3. Detect Key Change
        detected_key_info = self.key_detector.detect(midi_notes)
        if detected_key_info:
            detected_root, confidence = detected_key_info
            if detected_root != self.key:
                logger.info(f"[KEY DETECTOR] Changed Key: {self.key} -> {detected_root} ({confidence:.2f})")
                self.set_key(detected_root)
                
        # 4. Refine Prediction
        # 4.1. Get AI Distribution for next chord (precomputed)
        ai_probs = self.get_current_distribution()
        
        # 4.2. Get Ear Distribution for next chord (based on played notes)
        note_probs = self.ear.get_chord_probability_distribution(note_window)
        
        # 4.3. Combine distributions by multiplying probabilities (gets intersection)
        # TODO: try Weighted Sum - "Linear Opinion Pool"
        final_probs = {}
        all_chords = set(ai_probs.keys()) | set(note_probs.keys())
        
        for chord in all_chords:
            p_ai = ai_probs.get(chord, 0.001)
            if not note_probs:
                p_note = 1.0
            else:
                p_note = note_probs.get(chord, 0.001)
            
            final_probs[chord] = p_ai * p_note
            
        # Logging Top-5 for Debugging
        top_ai = get_top_k(ai_probs)
        top_ear = get_top_k(note_probs)
        top_final = get_top_k(final_probs)

        logger.info(f"[REFINEMENT] Step Combination:")
        logger.info(f"  AI (Chords):  {format_distribution(top_ai, self.key)}")
        logger.info(f"  EAR (Notes):  {format_distribution(top_ear, self.key)}")
        logger.info(f"  COMBINED:     {format_distribution(top_final, self.key)}\n")

        if not final_probs:
            return scheduled_chord

        # 5. Use argmax to get the best chord deterministically
        best_roman = max(final_probs, key=final_probs.get)
        best_prob = final_probs[best_roman]
        logger.info(f"[ARGMAX] Selected: {best_roman} (prob: {best_prob:.6f})")
        
        # 6. Create new Chord object
        root, chord_type = roman_to_chord(self.key, best_roman)
        final_chord_name = roman_to_compact(self.key, best_roman)
        logger.info(f"[ARGMAX] Converted to chord: {final_chord_name} = ({root}, {chord_type})")
        return Chord(root, chord_type, self.bpm, self.beats_per_bar)
