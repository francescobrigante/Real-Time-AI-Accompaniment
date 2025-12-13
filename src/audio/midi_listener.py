# =============================================================================
# MIDI Listener
# Listens to a virtual MIDI input port and forwards messages to SynthPlayer.
# Also captures notes for Harmony (Ear) and BPM detection.
# =============================================================================

import mido
import threading
from collections import deque
from typing import Optional, List, Tuple, Callable
import time

from src.audio.synth import SynthPlayer
from src.music.bpm_detector import BPMDetector
from src.config import HARMONY_WINDOW_SIZE, NOTES, DEFAULT_SOUNDFONT, DEFAULT_PROGRAM, DEFAULT_GAIN
from src.utils.logger import setup_logger

logger = setup_logger()

class MIDIListener:
    """
    Listens to a virtual MIDI input port and forwards messages to SynthPlayer.
    Runs in a separate thread for minimal latency.
    
    Architecture:
    - harmony_buffer: Stores notes for Ear/Predictor (preserved across tempo jumps)
    - bpm_detector: Receives onsets for tempo tracking (cleared on jumps)
    """
    
    def __init__(self, 
                 port_name: str, 
                 synth_player: SynthPlayer,
                 bpm_detector: Optional[BPMDetector] = None,
                 on_note_on: Optional[Callable[[int, float], None]] = None):

        self.port_name = port_name
        self.synth_player = synth_player
        self.bpm_detector = bpm_detector
        self.on_note_on = on_note_on                # User callback for note_on events
        
        self.is_running = False
        self.thread = None
        self.midi_port = None
        
        # Buffer for captured notes (note_val, timestamp)
        # Used by Predictor/Ear to analyze harmony. NOT cleared on tempo jumps.
        self.note_window = deque(maxlen=HARMONY_WINDOW_SIZE)
        
        self.lock = threading.RLock()
    
    # Internal listening loop running in separate thread
    def _listen_loop(self):
        try:
            self.midi_port = mido.open_input(self.port_name)
            # logger.info(f"[MIDI] Listening on {self.port_name}")
            
            while self.is_running:
                # Process all pending messages
                for msg in self.midi_port.iter_pending():
                    self._process_message(msg)
                
                # Sleep briefly to avoid 100% CPU usage
                time.sleep(0.001)
                
        except Exception as e:
            logger.error(f"[MIDI] Listener error: {e}")
            
        finally:
            # Cleanup on exit
            if self.midi_port:
                try:
                    self.midi_port.close()
                except:
                    pass
                logger.info("[MIDI] Port closed")
    
    def _process_message(self, msg):
        """Process a single MIDI message."""
        timestamp = time.time()
        
        # Handle note_on with velocity > 0 (actual note press)
        if msg.type == 'note_on' and msg.velocity > 0:
            midi_note = msg.note
            
            note_name = NOTES[midi_note % 12]
            octave = midi_note // 12 - 1
            
            # Add to harmony buffer (thread-safe)
            with self.lock:
                self.note_window.append((midi_note, timestamp))
            
            # Send to BPM detector
            if self.bpm_detector:
                self.bpm_detector.add_onset(timestamp)
            
            # User callback
            if self.on_note_on:
                try:
                    self.on_note_on(midi_note, timestamp)
                except Exception as e:
                    logger.error(f"[MIDI] Callback error: {e}")
        
        # Forward to synth immediately
        if self.synth_player:
            self.synth_player.handle_midi_message(msg)
    
    # Start listening in background thread
    def start(self):
        if self.is_running:
            logger.warning("[MIDI] Already listening")
            return
        
        self.is_running = True
        # execute listen loop on separate thread
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
    
    # Stop listening
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def get_note_window(self):
        """Returns the current window of captured notes."""
        with self.lock:
            return list(self.note_window)
        
    def clear_note_window(self):
        """Clears the harmony capture buffer."""
        with self.lock:
            self.note_window.clear()
            

# ================================= High-Level Interface =======================================

def create_playback_synth(
    midi_port_name: str, 
    soundfont_path: str = DEFAULT_SOUNDFONT, 
    program: int = DEFAULT_PROGRAM, 
    gain: float = DEFAULT_GAIN,
    bpm_detector: Optional[BPMDetector] = None
) -> tuple[SynthPlayer, MIDIListener]:
    """
    Create and initialize a complete real-time synth system.
    """
    
    # Create synth
    synth = SynthPlayer(soundfont_path=soundfont_path, program=program, gain=gain)
    
    # Initialize synth engine
    if not synth.initialize():
        raise RuntimeError("[ERROR] Failed to initialize synthesizer")
    
    # Create MIDI listener
    listener = MIDIListener(midi_port_name, synth, bpm_detector=bpm_detector)
    
    return synth, listener


# ----------------------------------- Main Test ---------------------------------------

def main():
    """Test the synth player with a simple sequence."""
    from src.config import INPUT_PORT
    logger.info("=" * 60)
    logger.info("FLUIDSYNTH REAL-TIME PLAYER TEST")
    logger.info("=" * 60)
        
    try:
        # Create synth system
        synth, listener = create_playback_synth(
            midi_port_name=INPUT_PORT,
            gain=1.0
        )
            
        logger.info(f"\nSynth ready! Listening on: {INPUT_PORT}")
        logger.info("Press Ctrl+C to stop...\n")
            
        listener.start()
            
        while True:
            time.sleep(0.1)
                
    except KeyboardInterrupt:
        logger.info("\nStopping...")
            
    except Exception as e:
        logger.error(f"\n{e}")
            
    finally:
        if 'listener' in locals():
            listener.stop()
        if 'synth' in locals():
            synth.cleanup()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()