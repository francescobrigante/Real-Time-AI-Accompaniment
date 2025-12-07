# =============================================================================
# Handles MIDI file generation and playback for Chord objects
# as MIDI events on a specified MIDI output port.
# =============================================================================

import mido
import time
from src.utils.logger import setup_logger

logger = setup_logger()

def play_chord(chord, output_port_name):
    """Plays a single Chord object on the specified MIDI port."""
    try:
        with mido.open_output(output_port_name) as out:
            start_time = time.time()
            for relative_time, msg in chord.midi_messages:
                target_time = start_time + relative_time
                current_time = time.time()
                wait_time = target_time - current_time
                if wait_time > 0:
                    time.sleep(wait_time)
                out.send(msg)
    except Exception as e:
        logger.error(f"Playback Error: {e}")

def play_chord_sequence(sequence, output_port_name):
    """Plays a sequence of Chord objects."""
    for i, chord in enumerate(sequence):
        logger.info(f"Playing chord {i+1}/{len(sequence)}: {chord}")
        play_chord(chord, output_port_name)
