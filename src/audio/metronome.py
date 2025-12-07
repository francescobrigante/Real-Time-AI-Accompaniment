# ==================================================================================================================
# Metronome Module
# Plays metronome clicks on each beat with an accent on the first beat of each bar
# ==================================================================================================================

import time
import threading
try:
    import fluidsynth
except ImportError:
    fluidsynth = None

from src.config import EMPTY_BARS_COUNT
from src.utils.logger import setup_logger
from src.audio.synth import start_audio_driver, load_soundfont

# Metronome configuration
METRONOME_NOTE = 76  # MIDI note for metronome click (E5)
METRONOME_VELOCITY = 100   # Velocity for metronome click, used to accent first beat
METRONOME_DURATION = 0.05  # Short click duration in seconds

# Synth-based metronome configuration
METRONOME_SOUNDFONT = "data/GeneralUser-GS.sf2"
METRONOME_PROGRAM = 115  # Woodblock (perfect for metronome)
METRONOME_CHANNEL = 9  # MIDI channel 9 is typically for percussion
METRONOME_GAIN = 0.8  # Volume level for metronome

logger = setup_logger()


class Metronome:
    """
    Handles metronome playback using FluidSynth.
    """
    
    def __init__(self, bpm: int, beats_per_bar: float, empty_bars_count: int = EMPTY_BARS_COUNT,
                 soundfont_path: str = METRONOME_SOUNDFONT, 
                 program: int = METRONOME_PROGRAM,
                 channel: int = METRONOME_CHANNEL,
                 gain: float = METRONOME_GAIN):
        
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.empty_bars_count = empty_bars_count
        self.channel = channel
        self.note = METRONOME_NOTE
        self.velocity = METRONOME_VELOCITY
        self.duration = METRONOME_DURATION
        
        self.synth = self._init_synth(soundfont_path, program, channel, gain)

    # Initialize FluidSynth
    def _init_synth(self, soundfont_path, program, channel, gain):
        if fluidsynth is None:
            return None
            
        try:
            # Create synth instance
            synth = fluidsynth.Synth(gain=gain, samplerate=44100.0)
            
            # Start audio driver (will raise RuntimeError if it fails)
            start_audio_driver(synth, "METRONOME")
            
            # Load SoundFont using shared helper
            sfid = load_soundfont(synth, soundfont_path)
            
            # Select metronome sound
            synth.program_select(channel, sfid, 0, program)
            
            # logger.info(f"[METRONOME] Synth initialized with program {program} on channel {channel}")
            return synth
            
        except Exception as e:
            logger.error(f"[METRONOME] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run(self, start_time_provider, is_running_provider, max_sequence_length):
        """
        Runs the metronome loop. Designed to be run in a separate thread.
        
        Args:
            start_time_provider: Lambda returning the start time (or None)
            is_running_provider: Lambda returning True if the pipeline is running
            max_sequence_length: Maximum number of chords to play
        """
        if not self.synth:
            logger.error("[METRONOME] Synth not initialized")
            return
        
        beat_duration = 60.0 / self.bpm
        
        # Wait for start time to be set from main thread
        while start_time_provider() is None and is_running_provider():
            time.sleep(0.01)
        
        if not is_running_provider():
            return
        
        # Get the actual start_time value
        start_time_value = start_time_provider()
        
        # Calculate total beats: empty bars + chord sequence
        delay_beats = int(self.empty_bars_count * self.beats_per_bar)
        total_beats = delay_beats + int(max_sequence_length * self.beats_per_bar)
        
        beat_count = 0
        while is_running_provider() and beat_count < total_beats:
            # Calculate when the next beat should occur
            beat_time = start_time_value + (beat_count * beat_duration)
            current_time = time.time()
            
            # Sleep until the next beat time
            wait_time = beat_time - current_time
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Accent first beat of each measure
            click_velocity = self.velocity + 20 if beat_count % self.beats_per_bar == 0 else self.velocity
            
            # Play click
            try:
                # Check if synth is still valid
                if self.synth:
                    self.synth.noteon(self.channel, self.note, click_velocity)
                    time.sleep(self.duration)
                    if self.synth:
                        self.synth.noteoff(self.channel, self.note)
            except Exception as e:
                logger.error(f"[METRONOME] Playback error: {e}")
            
            beat_count += 1
        
        logger.info("[METRONOME] Playback complete")

    def cleanup(self):
        """Stops and deletes the synth."""
        if self.synth:
            try:
                self.synth.delete()
            except Exception as e:
                logger.error(f"[METRONOME] Error during cleanup: {e}")
            self.synth = None

# ------------------------------------------ Main for testing ------------------------------------------

if __name__ == "__main__":
    metronome = Metronome(bpm=120, beats_per_bar=4)
    start_time = time.time() + 1
    is_running = True
    
    def get_start_time():
        return start_time
        
    def get_is_running():
        return is_running
        
    t = threading.Thread(target=metronome.run, args=(get_start_time, get_is_running, 4))
    t.start()
    
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        is_running = False
        t.join()
        metronome.cleanup()