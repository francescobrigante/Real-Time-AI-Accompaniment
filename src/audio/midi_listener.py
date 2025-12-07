# =============================================================================
# MIDI Listener
# Listens to a virtual MIDI input port and forwards messages to SynthPlayer.
# =============================================================================

import mido
import threading
from collections import deque
from src.audio.synth import SynthPlayer, DEFAULT_SOUNDFONT, DEFAULT_PROGRAM, DEFAULT_GAIN
from src.utils.logger import setup_logger
import time

logger = setup_logger()

class MIDIListener:
    """
    Listens to a virtual MIDI input port and forwards messages to SynthPlayer.
    Runs in a separate thread for minimal latency.
    """
    
    def __init__(self, port_name: str, synth_player: SynthPlayer):

        self.port_name = port_name                      # Name of MIDI input port to listen to
        self.synth_player = synth_player                # SynthPlayer instance to send messages to
        self.is_running = False                         # Whether the listener is running
        self.thread = None                              # Thread for running the listener
        self.midi_port = None                           # MIDI port for receiving messages
        
        # Buffer for captured notes (note_val, timestamp)
        # Used by Predictor/Ear to analyze harmony
        self.note_window = deque(maxlen=20)
    
    # Internal listening loop running in separate thread
    def _listen_loop(self):
        try:
            self.midi_port = mido.open_input(self.port_name)
            
            while self.is_running:
                # Process all pending messages
                for msg in self.midi_port.iter_pending():
                    # Capture note for prediction (Ear)
                    if msg.type == 'note_on' and msg.velocity > 0:
                        self.note_window.append((msg.note, time.time()))
                    
                    # Forward to synth immediately
                    self.synth_player.handle_midi_message(msg)
                
                # Sleep briefly to avoid 100% CPU usage
                time.sleep(0.001)
                
        except Exception as e:
            logger.error(f"[MIDI] Listener error: {e}")
            
        finally:
            # Cleanup on exit
            if self.midi_port:
                self.midi_port.close()
                logger.info("[MIDI] Port closed")
    
    
    # Start listening in background thread
    def start(self):

        if self.is_running:
            logger.warning("[MIDI] Already listening")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
    
    
    # Stop listening
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def get_note_window(self):
        """Returns the current window of captured notes."""
        return list(self.note_window)
        
    def clear_note_window(self):
        """Clears the capture buffer."""
        self.note_window.clear()
            

# ================================= High-Level Interface =======================================

def create_playback_synth(midi_port_name: str, soundfont_path: str = DEFAULT_SOUNDFONT, program: int = DEFAULT_PROGRAM, gain: float = DEFAULT_GAIN) -> tuple[SynthPlayer, MIDIListener]:
    """
    Create and initialize a complete real-time synth system.
    
    Args:
        midi_port_name: MIDI input port to listen to (e.g., 'IAC Piano IN')
        soundfont_path: Path to SoundFont file
        program: MIDI program number
        gain: Audio volume (0.0 to 1.0)
    
    Returns:
        Tuple of (SynthPlayer, MIDIListener) instances
    """
    
    # Create synth
    synth = SynthPlayer(soundfont_path=soundfont_path, program=program, gain=gain)
    
    # Initialize synth engine
    if not synth.initialize():
        raise RuntimeError("[ERROR] Failed to initialize synthesizer")
    
    # Create MIDI listener
    listener = MIDIListener(midi_port_name, synth)
    
    return synth, listener


# ================================= Main Test =======================================

def main():
    """Test the synth player with a simple sequence."""
    
    # Configuration
    MIDI_PORT = 'IAC Piano IN'
    SOUNDFONT = DEFAULT_SOUNDFONT
    PROGRAM = 0
    GAIN = 1.0
    
    logger.info("=" * 60)
    logger.info("FLUIDSYNTH REAL-TIME PLAYER TEST")
    logger.info("=" * 60)
    
    # Check audio setup
    try:
        import subprocess
        result = subprocess.run(['system_profiler', 'SPAudioDataType'], capture_output=True, text=True, timeout=5)
        logger.debug("Audio devices found")
    except:
        logger.debug("Could not check audio devices")
    
    print()
    
    try:
        # Create synth system
        synth, listener = create_playback_synth(
            midi_port_name=MIDI_PORT,
            soundfont_path=SOUNDFONT,
            program=PROGRAM,
            gain=GAIN
        )
        
        logger.info(f"\nSynth ready! Listening on: {MIDI_PORT}")
        logger.info("Press Ctrl+C to stop...\n")
        
        # Start listening
        listener.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("\nStopping...")
        
    except Exception as e:
        logger.error(f"\n{e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if 'listener' in locals():
            listener.stop()
        if 'synth' in locals():
            synth.cleanup()
        
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()