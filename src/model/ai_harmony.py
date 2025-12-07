# ===========================================================================
# AIHarmonyRules Class for LSTM-based chord prediction
#
#  Uses a pre-trained LSTM model to predict the next chord given a history
#  Pre-computes sequences of chords for efficiency, adds dynamic PADDING
#  if history is shorter than expected, and supports both deterministic
#  (argmax) and probabilistic sampling.
# ===========================================================================

import torch
import pickle
import random
import os
import numpy as np
from typing import List, Tuple, Dict
from src.utils.music_theory import roman_to_chord, compact_chord
from src.model.lstm_model import ChordLSTM
from src.config import (
    MODEL_PATH, VOCAB_PATH, 
    HIDDEN_SIZE, EMBEDDING_DIM, NUM_LAYERS, WINDOW_SIZE, DROPOUT, 
    PAD_IDX, UNKNOWN_IDX
)
from src.utils.logger import setup_logger

logger = setup_logger()

class AIHarmonyRules:
    def __init__(self, key: str = 'C'):
        self.key = key
        self.device = self._get_device()
        
        # Load vocab
        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(f"Vocabulary not found at {VOCAB_PATH}. Run training first.")
            
        with open(VOCAB_PATH, 'rb') as f:
            data = pickle.load(f)
            self.chord_to_idx = data['chord_to_idx']
            self.idx_to_chord = data['idx_to_chord']
            
        self.vocab_size = len(self.chord_to_idx)
        
        # Load model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
            
        self.model = ChordLSTM(self.vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, padding_idx=PAD_IDX)
        
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        except RuntimeError as e:
            logger.error(f"\n[LSTM LOADING ERROR] Model shape mismatch! You changed hyperparameters but didn't re-train the model.")
            logger.error(f"Expected: Hidden={HIDDEN_SIZE}, Layers={NUM_LAYERS}, Embed={EMBEDDING_DIM}")
            raise e
            
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"[LSTM INFO] Loaded LSTM model from {MODEL_PATH}")
        logger.info(f"[LSTM INFO] Using device: {self.device}")
        
    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
        
    def set_key(self, new_key: str) -> bool:
        self.key = new_key
        return True
       
    # Predict next chord distribution using LSTM-based model
    # Can use both deterministic (argmax) and probabilistic sampling
    # Returns either roman numeral or (root, type) tuple based on return_roman flag 
    def get_next_chord_distribution(self, chord_history: List[str], return_roman: bool = False, deterministic_sampling: bool = True, temperature: float = 1.0, top_k: int = 0) -> Tuple[str, Dict[str, float]]:

        if not chord_history:
            # Fallback for empty history
            logger.info("[LSTM INFO] Empty chord history, using fallback chord 'I'")
            return ('I', {'I': 1.0}) if return_roman else (('C', 'major'), {'I': 1.0})

        # Get last WINDOW_SIZE chords
        current_history = list(chord_history)
        if len(current_history) > WINDOW_SIZE:
            current_history = current_history[-WINDOW_SIZE:]
        
        # Convert to indices
        input_indices = [self.chord_to_idx.get(c, UNKNOWN_IDX) for c in current_history]
        
        # Pad at the beginning if shorter than WINDOW_SIZE
        if len(input_indices) < WINDOW_SIZE:
            padding_needed = WINDOW_SIZE - len(input_indices)
            input_indices = [PAD_IDX] * padding_needed + input_indices
            
        # Create tensor directly on device
        input_tensor = torch.tensor([input_indices], dtype=torch.long, device=self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            # Apply temperature
            if not deterministic_sampling and temperature != 1.0:
                 output = output / temperature
            probs = torch.softmax(output, dim=1)
            
        # Sample from distribution
        probs_np = probs.cpu().numpy()[0]
        
        # Safety check for NaN
        if torch.isnan(probs).any():
             logger.warning("[AI WARNING] NaN in probabilities, using uniform distribution")
             probs_np = np.ones(self.vocab_size) / self.vocab_size

        if deterministic_sampling:
            # Argmax: Choose the index with the highest probability
            # mask out padding
            probs_np[PAD_IDX] = -1.0 
            predicted_idx = np.argmax(probs_np)
        else:
            # Weighted Random Sampling
            # Ensure PAD_IDX is not sampled by setting its probability to 0
            probs_np[PAD_IDX] = 0
            
            # --- TOP-K SAMPLING ---
            if top_k > 0:
                # Get indices of top k elements
                # Use argpartition for efficiency (O(n)) vs sort (O(n log n))
                if top_k < len(probs_np):
                    ind = np.argpartition(probs_np, -top_k)[-top_k:]
                    
                    # Create a mask for zeroing out everything else
                    mask = np.zeros_like(probs_np, dtype=bool)
                    mask[ind] = True
                    
                    # Apply mask
                    probs_np[~mask] = 0.0

            # Re-normalize probabilities 
            if probs_np.sum() > 0:
                probs_np = probs_np / probs_np.sum()
            else: # Fallback if all other probabilities are zero
                probs_np = np.ones(self.vocab_size) / self.vocab_size
                probs_np[PAD_IDX] = 0
                probs_np = probs_np / probs_np.sum() # Re-normalize again

            predicted_idx = random.choices(range(self.vocab_size), weights=probs_np)[0]

        predicted_roman = self.idx_to_chord[predicted_idx]
        
        # Create probability dictionary
        prob_dict = {}
        for idx, prob in enumerate(probs_np):
            if idx == PAD_IDX: continue # Don't return probability for PAD
            # Only return non-zero probabilities if top-k is used to keep dict clean
            if top_k > 0 and prob == 0: continue
            
            chord_name = self.idx_to_chord[idx]
            prob_dict[chord_name] = float(prob)
            
        if return_roman:
            return predicted_roman, prob_dict
        else:
            root, chord_type = roman_to_chord(self.key, predicted_roman)
            
            # Convert roman probabilities to compact chord string probabilities
            probabilities_string = {}
            for roman, prob in prob_dict.items():
                c_root, c_quality = roman_to_chord(self.key, roman)
                c_string = compact_chord(c_root, c_quality)
                probabilities_string[c_string] = prob
            
            return (root, chord_type), probabilities_string

    # precomputes a sequence of chords based on argmax prediction
    # returns list of (roman_numeral, probability_dict) tuples
    def precompute_sequence(self, start_history: List[str], length: int, 
                            deterministic_sampling: bool = True,
                            temperature: float = 1.0,
                            top_k: int = 0) -> List[Tuple[str, Dict[str, float]]]:

        sequence = []
        current_history = list(start_history)
        
        for _ in range(length):
            # Predict next chord (deterministic)
            predicted_roman, prob_dict = self.get_next_chord_distribution(
                current_history, 
                return_roman=True, 
                deterministic_sampling=deterministic_sampling,
                temperature=temperature,
                top_k=top_k
            )
            
            sequence.append((predicted_roman, prob_dict))
            current_history.append(predicted_roman)
            
        return sequence

if __name__ == "__main__":
    # Test
    try:
        ai = AIHarmonyRules('A')
        history = ['I', 'vi7', 'ii', 'V7']
        logger.info(f"History: {history}")
        prediction, probs = ai.get_next_chord_distribution(history, return_roman=True)
        logger.info(f"Prediction: {prediction}")
        
        logger.info("\nPrecomputing sequence of 4:")
        seq = ai.precompute_sequence(history, 4)
        for i, (chord, _) in enumerate(seq):
            logger.info(f"  Step {i+1}: {chord}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
