# ==================================================================================================
# CONFIGURATION & CONSTANTS
# ==================================================================================================

import numpy as np

# ------------------------------------------------------------------
# MIDI PORTS
# ------------------------------------------------------------------
OUTPUT_PORT = 'IAC Piano IN'  # vmpk virtual piano for playback
INPUT_PORT = 'IAC Piano OUT'  # vmpk virtual piano for notes input
# INPUT_PORT = 'Digital Piano' # yamaha physical keyboard

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
DATA_DIR = 'data'
RAW_DATA_PATH = 'data/chordonomicon_v2.csv'
CLEAN_DATA_PKL = 'data/clean_dataset.pkl'
CLEAN_DATA_CSV = 'data/clean_dataset.csv'
MODEL_PATH = "data/best_model.pth"
VOCAB_PATH = "data/vocab.pkl"
TEST_SET_PATH = "data/test_set.pkl"
PLOT_PATH = "data/training_plot.png"

# ------------------------------------------------------------------
# MODEL HYPERPARAMETERS
# ------------------------------------------------------------------
WINDOW_SIZE = 8        # Number of chords to consider for prediction
HIDDEN_SIZE = 512
EMBEDDING_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.3
PAD_IDX = 0            # Vocabulary index for padding
UNKNOWN_IDX = 1        # Vocabulary index for unknown chords

# ------------------------------------------------------------------
# TRAINING CONFIGURATION
# ------------------------------------------------------------------
BATCH_SIZE = 1024
EPOCHS = 8
LEARNING_RATE = 0.0002
NUM_WORKERS = 2

# ------------------------------------------------------------------
# TIMING & PIPELINE
# ------------------------------------------------------------------
DEFAULT_BPM = 120
DEFAULT_BEATS_PER_BAR = 4.0
DEFAULT_MAX_SEQUENCE_LENGTH = 10

DELAY_START_SECONDS = 2.0
EMPTY_BARS_COUNT = 1  # Default empty bars for metronome
CHORDS_TO_PRECOMPUTE = 10 # Number of chords to precompute for the LSTM

# ------------------------------------------------------------------
# PREDICTION & SAMPLING
# ------------------------------------------------------------------
EXPONENTIAL_WEIGHT_FACTOR = 0.3 # Exponential weight factor for note weighting in ear module. higher = more emphasis on recent notes
AI_WEIGHT = 0.2                # Weight for LSTM in final prediction (1.0 = AI only, 0.0 = Ear only)
USE_DETERMINISTIC_SAMPLING = False
SAMPLING_TOP_K = 85             # Top-K sampling for LSTM
SAMPLING_TEMPERATURE = 0.13     # Temperature for LSTM sampling (higher = more random)

# ------------------------------------------------------------------
# MUSIC THEORY MAPPINGS
# ------------------------------------------------------------------

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11]  # Semitones from root

NOTE_TO_MIDI_MAP = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11
}

FLAT_TO_SHARP = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'Cb': 'B', 'Fb': 'E'
}

# Maps intervals strings to intervals numbers
INTERVALS_MAP = {
    'major': [0, 4, 7],        # Major Triad
    'minor': [0, 3, 7],        # Minor Triad
    '7': [0, 4, 7, 10],        # Dominant 7th
    'm7': [0, 3, 7, 10],       # Minor 7th
    'maj7': [0, 4, 7, 11],     # Major 7th
    'dim': [0, 3, 6],          # Diminished Triad
    'aug': [0, 4, 8],          # Augmented Triad
    'sus2': [0, 2, 7],         # Suspended 2nd
    'sus4': [0, 5, 7],         # Suspended 4th
    '6': [0, 4, 7, 9],         # Major 6th
    'minor6': [0, 3, 7, 9],    # Minor 6th
    '9': [0, 4, 7, 10, 14],    # Dominant 9th
    'add9': [0, 4, 7, 14],     # Add 9th
    'dim7': [0, 3, 6, 9],      # Diminished 7th
    'm7b5': [0, 3, 6, 10]      # Half-Diminished 7th
}

# Chromatic Roman to Semitone Map (Trailing Accidentals)
ROMAN_TO_SEMITONE = {
    'I': 0, 'i': 0,
    'IIb': 1, 'iib': 1, 'I#': 1, 'i#': 1,
    'II': 2, 'ii': 2,
    'IIIb': 3, 'iiib': 3, 'II#': 3, 'ii#': 3,
    'III': 4, 'iii': 4,
    'IV': 5, 'iv': 5,
    'IV#': 6, 'iv#': 6, 'Vb': 6, 'vb': 6,
    'V': 7, 'v': 7,
    'VIb': 8, 'vib': 8, 'V#': 8, 'v#': 8,
    'VI': 9, 'vi': 9,
    'VIIb': 10, 'viib': 10, 'VI#': 10, 'vi#': 10,
    'VII': 11, 'vii': 11
}

# Map semitone intervals to Roman Numerals (Chromatic - Trailing)
INTERVAL_TO_ROMAN = {
    0: 'I', 1: 'IIb', 2: 'II', 3: 'IIIb', 4: 'III', 5: 'IV',
    6: 'IV#', 7: 'V', 8: 'VIb', 9: 'VI', 10: 'VIIb', 11: 'VII'
}

# Harmonic roles and transitions
DEGREE_TO_ROLE = {
    0: 'T', 1: 'S', 2: 'T', 3: 'S', 4: 'T', 5: 'S',
    6: 'D', 7: 'D', 8: 'D', 9: 'T', 10: 'D', 11: 'D'
}

TONIC_CHORDS = ['I', 'vi', 'iii', 'Imaj7', 'I7', 'vi7', 'iii7', 'i', 'i7']
SUBDOMINANT_CHORDS = ['IV', 'ii', 'IVmaj7', 'IV7', 'ii7', 'iv', 'iv7', 'II', 'II7']
DOMINANT_CHORDS = ['V', 'vii°', 'V7', 'vii°7', 'III', 'III7', 'v', 'v7']

CHORD_ROLES = {
    'T': TONIC_CHORDS,
    'S': SUBDOMINANT_CHORDS,
    'D': DOMINANT_CHORDS
}

ROLE_TRANSITIONS = {
    'T': [('S', 0.45), ('D', 0.35), ('T', 0.20)],
    'S': [('D', 0.50), ('T', 0.30), ('S', 0.20)],
    'D': [('T', 0.65), ('S', 0.25), ('D', 0.10)]
}

# Krumhansl-Schmuckler Major Profile (Enhanced)
MAJOR_PROFILE = np.array([
    6.50, 1.80, 3.80, 1.80, 5.00, 4.50, 1.80, 5.60, 1.80, 4.00, 1.80, 3.40
], dtype=np.float32)
