# ==================================================================================================================
#                               Ear (Note-Based Harmony Prediction Engine)
# ==================================================================================================================
# Analyzes played notes to predict next chord using functional harmony theory (T-S-D roles)
# Uses exponential weighting to prioritize recent notes over older ones
# ==================================================================================================================

import random
from typing import List, Tuple, Dict, Optional
from math import exp

from src.config import NOTES, EXPONENTIAL_WEIGHT_FACTOR, DEGREE_TO_ROLE, CHORD_ROLES, ROLE_TRANSITIONS
from src.utils.logger import setup_logger

logger = setup_logger()


class Ear:
    
    def __init__(self, key: str = 'C'):
        try:
            self.key = key
            self.key_index = NOTES.index(key)
            
        except ValueError:
            logger.warning(f"Key '{key}' not recognized. Defaulting to 'C'.")
            self.key = 'C'
            self.key_index = 0
     
    def set_key(self, new_key: str) -> bool:
        try:
            new_key_index = NOTES.index(new_key)
            self.key = new_key
            self.key_index = new_key_index
            return True
        except ValueError:
            logger.warning(f"Key '{new_key}' not recognized. Key unchanged.")
            return False
    
    
    # Classify a single MIDI note number (0-127) as T (tonic), S (subdominant), or D (dominant)
    def _classify_note(self, midi_note: int) -> str:
        # Get note relative to key (scale degree 0-11)
        note_in_scale = midi_note % 12
        degree = (note_in_scale - self.key_index) % 12
        
        return DEGREE_TO_ROLE[degree] # returns 'T', 'S', or 'D'
    
    
    def _compute_exponential_weights(self, window_size: int, alpha: float = EXPONENTIAL_WEIGHT_FACTOR) -> List[float]:
        """
        Generate exponential weights for note window: recent notes get higher weight.
        Uses exponential decay formula: weight[i] = exp(alpha * i)
        where i goes from 0 (oldest) to window_size-1 (newest)
        Exponential growth factor: higher = more emphasis on recent notes and less on old ones
        """
        if window_size == 0:
            return []
        
        if window_size == 1:
            return [1.0]
        
        weights = []
        for i in range(window_size):
            weight = exp(alpha * i)  # e^(alpha * i)
            weights.append(weight)
        
        # Normalize
        total = sum(weights)
        normalized = [w / total for w in weights]
        
        return normalized
    
    
    # Given window of notes, compute weighted scores for T, S, D, giving more weight to recent notes
    def _compute_window_scores(self, note_window: List[Tuple[int, float]]) -> Dict[str, float]:
        """
        Returns:
            Dictionary {'T': score, 'S': score, 'D': score}
        """
        scores = {'T': 0.0, 'S': 0.0, 'D': 0.0}
        
        if not note_window:
            return scores
        
        # Get exponential weights
        weights = self._compute_exponential_weights(len(note_window), alpha=EXPONENTIAL_WEIGHT_FACTOR)
        
        # Classify each note and add weighted score
        for i, (midi_note, duration) in enumerate(note_window):
            role = self._classify_note(midi_note)
            scores[role] += weights[i]
        
        return scores
    
    # Computes the probability distribution for the next chord based on the played notes
    # Uses role transitions (T->S->D pattern) to guide harmonic progression
    def get_chord_probability_distribution(self, note_window: List[Tuple[int, float]]) -> Dict[str, float]:

        # 1. Calculate role scores from played notes
        role_scores = self._compute_window_scores(note_window)
        if sum(role_scores.values()) == 0:
            return {}
        
        # 2. Determine current role and use weighted sampling for next role
        max_window_role = max(role_scores, key=role_scores.get)
        
        note_names = [NOTES[n[0] % 12] for n in note_window]
        
        if max_window_role not in ROLE_TRANSITIONS:
            transitions = [('T', 0.33), ('S', 0.33), ('D', 0.34)]
        else:
            transitions = ROLE_TRANSITIONS[max_window_role]
        
        # Weighted sampling: choose next role probabilistically
        roles, weights = zip(*transitions)
        chosen_role = random.choices(roles, weights=weights)[0]
        
        logger.info("-" * 60 + "[PREDICTION]" + "-" * 60)
        logger.info(f"[EAR] Note window: {note_names} | Scores: T={role_scores['T']:.3f}, S={role_scores['S']:.3f}, D={role_scores['D']:.3f} | From {max_window_role} to {chosen_role}")
        
        # 3. Distribute: 90% to chosen role, 10% to others
        chord_probs = {}
        for role in ['T', 'S', 'D']:
            chords = CHORD_ROLES[role]
            weight = 0.90 if role == chosen_role else 0.05
            prob_per_chord = weight / len(chords)
            for chord in chords:
                chord_probs[chord] = prob_per_chord
        
        return chord_probs
    
    
    

# ================================= Main test block  =========================================

if __name__ == "__main__":
    
    key = 'C'
    predictor = Ear(key)

    # Test cases: (description, note_window)
    test_cases = [
        (
            "Test 1: Mixed sequence (T -> S -> D)",
            [
                (60, 1.0),  # C (T) - oldest
                (64, 1.0),  # E (T)
                (62, 1.0),  # D (S)
                (65, 1.0),  # F (S)
                (67, 1.5),  # G (D) - newest, highest weight
                (71, 1.5),  # B (D) - newest, highest weight
            ]
        ),
        (
            "Test 2: I chord (C major: C, E, G) -> should go to S or D",
            [(60, 1.0), (64, 1.0), (67, 1.0)]  # C, E, G
        ),
        (
            "Test 3: IV chord (F major: F, A, C) -> should often go to V or I",
            [(65, 1.0), (69, 1.0), (60, 1.0)]  # F, A, C
        ),
        (
            "Test 4: V chord (G major: G, B, D) -> should strongly resolve to I",
            [(67, 1.0), (71, 1.0), (62, 1.0)]  # G, B, D
        ),
        (
            "Test 5: ii chord (D minor: D, F, A) -> should go to V",
            [(62, 1.0), (65, 1.0), (69, 1.0)]  # D, F, A
        ),
        (
            "Test 6: vi chord (A minor: A, C, E) -> should go to IV or ii",
            [(69, 1.0), (60, 1.0), (64, 1.0)]  # A, C, E
        )
    ]
    
    # Run all tests using probability distribution
    for description, note_window in test_cases:
        print(f"\n\n====== {description}")
        dist = predictor.get_chord_probability_distribution(note_window)
        scores = predictor._compute_window_scores(note_window)

        if not dist:
            logger.info("No distribution (empty window)")
            continue

        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        topk = sorted_dist[:10]
        logger.info(f"[TEST] Top chords (roman -> prob): {topk}")
