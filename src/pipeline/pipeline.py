# ==================================================================================================================
# Real-Time Accompaniment Pipeline
# ==================================================================================================================
# Orchestrates the real-time accompaniment pipeline
# Contains timing thread which is the main thread for prediction and timing alignment
# Has start() function which is the entry point for the pipeline
# ==================================================================================================================

import time
import threading
from typing import List, Optional

from src.config import (
    OUTPUT_PORT, DELAY_START_SECONDS, EMPTY_BARS_COUNT,
    DEFAULT_BPM, DEFAULT_BEATS_PER_BAR, WINDOW_SIZE, 
    DEFAULT_MAX_SEQUENCE_LENGTH, CHORDS_TO_PRECOMPUTE
)
from src.utils.logger import setup_logger
from src.utils.music_theory import compact_chord, chord_to_roman
from src.music.chord import Chord

from src.pipeline.predictor import Predictor
from src.pipeline.playback import PlaybackThread
from src.audio.metronome import Metronome
from src.audio.midi_listener import MIDIListener
from src.audio.midi_listener import create_playback_synth

logger = setup_logger()

class RealTimePipeline:
    
    def __init__(self, key: str = 'C', chord_type: str = 'major', bpm: int = DEFAULT_BPM, 
                 beats_per_bar: float = DEFAULT_BEATS_PER_BAR, window_size: int = WINDOW_SIZE, 
                 max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH, output_port: str = OUTPUT_PORT, 
                 input_port: Optional[str] = None, enable_input_listener: bool = False, 
                 enable_metronome: bool = True, empty_bars_count: int = EMPTY_BARS_COUNT,
                 enable_synth: bool = True):
        
        # Configuration
        self.key = key
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.chord_duration_seconds = (beats_per_bar * 60.0) / bpm
        self.window_size = window_size
        self.max_sequence_length = max_sequence_length
        self.output_port = output_port
        self.input_port = input_port
        self.enable_input_listener = enable_input_listener
        self.enable_metronome = enable_metronome
        self.empty_bars_count = empty_bars_count
        self.enable_synth = enable_synth
        
        # Delay calculation: how many seconds to wait before starting the sequence
        if self.enable_metronome:
            self.delay_seconds = empty_bars_count * beats_per_bar * (60.0 / bpm)
        else:
            self.delay_seconds = DELAY_START_SECONDS
            
        # Predictor (Predicts the next chord)
        self.predictor = Predictor(key, bpm, beats_per_bar, window_size)
            
        # FluidSynth (for Playback)
        self.synth = None
        self.synth_listener = None
        if self.enable_synth and self.output_port:
            try:
                self.synth, self.synth_listener = create_playback_synth(midi_port_name=self.output_port)
                logger.info(f"[SYNTH] FluidSynth initialized on {self.output_port}")
            except Exception as e:
                logger.error(f"[SYNTH] Failed to initialize FluidSynth: {e}")

        # MIDI Input Listener (for Ear)
        if self.enable_input_listener and self.input_port:
            self.midi_listener = MIDIListener(port_name=self.input_port, synth_player=self.synth)
        else:
            self.midi_listener = None
        
        # Metronome
        self.metronome = Metronome(self.bpm, self.beats_per_bar, self.empty_bars_count)
        self.metronome_thread = None
        self.timing_thread = None
        self.playback = None
        
        # State
        self.chord_objects = []                     # Shared list containing the chords to play
        self.current_chord_idx = 0                  # Current chord index
        self.is_running = False                     # Pipeline running flag
        self.start_time = None                      # Pipeline start time
        
        # Starting Chord
        self.starting_chord = Chord(key, chord_type, bpm, beats_per_bar)

    # MAIN THREAD: handles all chord scheduling, prediction and timing alignment
    def _timing_thread(self):
        
        # Get starting chord roman
        start_roman = chord_to_roman(self.key, self.starting_chord.root, self.starting_chord.chord_type)
        
        # First thing to do is to seed the window and compute the initial sequence with LSTM
        self.predictor.chord_window.append(start_roman)
        self.predictor.precomputed_sequence = self.predictor.ai.precompute_sequence(list(self.predictor.chord_window), CHORDS_TO_PRECOMPUTE)
        self.predictor.precomputed_idx = 0

        # Only then start the timer to signal the start of the pipeline
        self.start_time = time.time()
        
        # Wait for delay
        time.sleep(self.delay_seconds)
        
        # Play starting chord
        self.chord_objects.append(self.starting_chord)
        
        # Then set the current chord index to 1
        self.current_chord_idx = 1
        logger.info(f"[{time.time() - self.start_time:.1f}s][🎶PLAYING STEP 1🎶] {self.starting_chord}")
        
        while self.is_running and self.current_chord_idx < self.max_sequence_length:
            # Timing
            next_chord_time = self.start_time + self.delay_seconds + (self.current_chord_idx * self.chord_duration_seconds)
            wait_time = next_chord_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
                
            # Prediction
            # 1. Get next candidate from Predictor (Precomputed AI)
            candidate_chord = self.predictor.get_next_prediction()
            
            if not candidate_chord:
                logger.warning("No prediction available, ending sequence.")
                break
                
            # 2. Refine with Ear
            final_chord = self.predictor.refine_prediction(candidate_chord, self.midi_listener)
            
            # WAITING LOGIC: if no notes played, wait for input
            # In the meantime, keep time going 
            if not final_chord:
                logger.info("[SYSTEM] No notes played, waiting...")
                self.current_chord_idx += 1
                if self.midi_listener: self.midi_listener.clear_note_window()
                continue
                
            # 3. Schedule final chord
            self.chord_objects.append(final_chord)
            
            # 4. Sync key with predictor (in case key detector changed it)
            if self.key != self.predictor.key:
                self.key = self.predictor.key
            
            # 5. Update History (Important for re-computation!)
            # If the final chord matches the one in the history, then go to next
            # Otherwise, update history by pre-computing again
            final_roman = chord_to_roman(self.key, final_chord.root, final_chord.chord_type)
            self.predictor.update_history(final_roman)
            
            self.current_chord_idx += 1
            logger.info(f"[{time.time() - self.start_time:.1f}s][🎶PLAYING STEP {self.current_chord_idx}🎶] {final_chord}")
            
            if self.midi_listener: self.midi_listener.clear_note_window()
            
        logger.info("[SYSTEM] Sequence complete!")
        self.is_running = False

    # REAL-TIME PIPELINE ENTRY POINT: Orchestrates all the threads
    def start(self):
        if self.is_running: return
        
        print()
        logger.info(f"[SYSTEM] Starting pipeline in {self.key} @ {self.bpm} BPM")
        self.is_running = True
        
        # Start Listeners
        if self.midi_listener: self.midi_listener.start()
        if self.synth_listener: self.synth_listener.start()
        
        # Threads
        threads = []
        
        # 1. Metronome
        if self.enable_metronome:
            self.metronome_thread = threading.Thread(
                target=self.metronome.run,
                args=(lambda: self.start_time, lambda: self.is_running, self.max_sequence_length),
                daemon=True,
            )
            self.metronome_thread.start()
            threads.append(self.metronome_thread)
            
        # 2. Timing (Predictor)
        self.timing_thread = threading.Thread(target=self._timing_thread, daemon=True)
        self.timing_thread.start()
        threads.append(self.timing_thread)
        
        # 3. Playback (Synth for output)
        self.playback = PlaybackThread(self.chord_objects, lambda: self.start_time, 
                           self.delay_seconds, self.chord_duration_seconds, self.output_port,
                           self.max_sequence_length, lambda: self.is_running)
        self.playback.start()
        threads.append(self.playback)
        
        # Wait for threads to finish
        # Wait for threads to finish
        try:
            while True:
                threads_alive = False
                if self.timing_thread and self.timing_thread.is_alive():
                    self.timing_thread.join(timeout=0.1)
                    threads_alive = True
                if self.playback and self.playback.is_alive():
                    self.playback.join(timeout=0.1)
                    threads_alive = True
                
                if not threads_alive:
                    break

        except KeyboardInterrupt:
            logger.warning("[SYSTEM] Stopping pipeline...")
        finally:
            self.stop()
            
        return self.chord_objects

    # Stop the pipeline
    def stop(self):
        self.is_running = False

        # Stop playback first so threads can exit promptly
        if self.playback: self.playback.stop()

        # Stop listeners / synth
        if self.midi_listener: self.midi_listener.stop()
        if self.synth_listener: self.synth_listener.stop()
        if self.synth: self.synth.cleanup()

        # Join helper threads if still alive (best-effort, short wait)
        if self.playback and self.playback.is_alive():
            self.playback.join(timeout=1.0)
        if self.metronome_thread and self.metronome_thread.is_alive():
            self.metronome_thread.join(timeout=1.0)
        if self.timing_thread and self.timing_thread.is_alive():
            self.timing_thread.join(timeout=1.0)

        # Shutdown predictor executor to avoid stray threads
        try:
            self.predictor.close()
        except Exception as e:
            logger.error(f"[SYSTEM] Predictor shutdown error: {e}")

    # Get the current sequence of chords
    def get_current_sequence(self) -> List[str]:
        if not self.chord_objects: return []
        return [compact_chord(c.root, c.chord_type) for c in self.chord_objects]