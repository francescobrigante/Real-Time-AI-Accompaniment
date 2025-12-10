# =============================================================================
# FluidSynth MIDI Playback Module
# Low-latency modular synthesizer for real-time accompaniment playback
# =============================================================================

import mido
from typing import Optional
from src.config import NOTES
from src.utils.logger import setup_logger

logger = setup_logger()

try:
    import fluidsynth
    FLUIDSYNTH_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    logger.error(f"FluidSynth not available: {e}")
    FLUIDSYNTH_AVAILABLE = False
    fluidsynth = None

# Static data
DEFAULT_SOUNDFONT = "data/GeneralUser-GS.sf2"           # Sounds used for synthesis
DEFAULT_PROGRAM = 0                                     # 0 = Acoustic Grand Piano
DEFAULT_CHANNEL = 0                                     # MIDI channel to use (0-15)
DEFAULT_GAIN = 1.0                                      # Volume level

# Audio driver selection based on OS (single preferred driver per platform)
AUDIO_DRIVERS = {
    'win32': 'dsound',      # Windows (wasapi can be noisy with warnings)
    'darwin': 'coreaudio',  # macOS
    'linux': 'alsa'         # Linux
}


# ================================= Helper Functions =======================================

def get_platform_audio_driver():
    """Get the preferred audio driver for the current platform."""
    import sys
    return AUDIO_DRIVERS.get(sys.platform, 'coreaudio')


def start_audio_driver(synth, logger_prefix="SYNTH"):
    """
    Start audio driver for the current platform. Fails immediately if driver cannot start.
    
    Args:
        synth: FluidSynth instance
        logger_prefix: Prefix for log messages
    
    Returns:
        True if driver started successfully
    
    Raises:
        RuntimeError: If the preferred audio driver fails to start
    """
    driver = get_platform_audio_driver()
    
    try:
        logger.info(f"[{logger_prefix}] Starting audio driver: {driver}")
        synth.start(driver=driver)
        logger.info(f"[{logger_prefix}] Audio driver '{driver}' started successfully")
        return True
    except Exception as e:
        error_msg = f"[{logger_prefix}] Failed to start audio driver '{driver}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def load_soundfont(synth, soundfont_path):
    """
    Load a SoundFont file into the synthesizer.
    
    Args:
        synth: FluidSynth instance
        soundfont_path: Path to .sf2 file
    
    Returns:
        SoundFont ID on success, -1 on failure
    """
    sfid = synth.sfload(soundfont_path)
    if sfid == -1:
        raise FileNotFoundError(f"Cannot load SoundFont: {soundfont_path}")
    return sfid


# ================================= Core Synth Player class =======================================

class SynthPlayer:
    
    def __init__(self, soundfont_path: str = DEFAULT_SOUNDFONT, 
                 program: int = DEFAULT_PROGRAM, 
                 channel: int = DEFAULT_CHANNEL,
                 gain: float = DEFAULT_GAIN,
                 audio_driver: Optional[str] = None):
        """
        Args:
            soundfont_path: Path to .sf2 SoundFont file
            program: MIDI program number (0 = Acoustic Grand Piano)
            channel: MIDI channel to use (0-15)
            gain: Audio gain/volume (0.0 to 1.0)
            audio_driver: Audio driver ('coreaudio', 'alsa', 'dsound'). Auto-detected if None.
        """

        self.soundfont_path = soundfont_path
        self.program = program
        self.channel = channel
        self.gain = gain
        
        # Auto-detect audio driver if not specified
        if audio_driver is None:
            import sys
            platform = sys.platform
            self.audio_driver = AUDIO_DRIVERS.get(platform, 'coreaudio')
        else:
            self.audio_driver = audio_driver
        logger.info(f"[SYNTH] Using audio driver: {self.audio_driver}")
        
        # Synth state
        self.synth = None
        self.sfid = None
        self.is_running = False
        
    
    def initialize(self) -> bool:
        """
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create synth instance
            self.synth = fluidsynth.Synth(gain=self.gain, samplerate=44100.0)
            
            # Start audio driver (will raise RuntimeError if it fails)
            start_audio_driver(self.synth, "SYNTH")
            
            # Load SoundFont
            self.sfid = load_soundfont(self.synth, self.soundfont_path)
            self.synth.program_select(self.channel, self.sfid, 0, self.program)
            self.is_running = True
            
            return True
            
        except Exception as e:
            logger.error(f"[SYNTH] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup()
            
            return False
    
    
    def play_note(self, note: int, velocity: int = 100):
        """
        Args:
            note: MIDI note number (0-127)
            velocity: Note velocity (0-127)
        """
        if self.synth and self.is_running:
            self.synth.noteon(self.channel, note, velocity)
    
    
    def stop_note(self, note: int):
        if self.synth and self.is_running:
            self.synth.noteoff(self.channel, note)
    
    
    def handle_midi_message(self, msg: mido.Message):
        """
        Process a MIDI message and send to synth for playback
        
        Args:
            msg: mido.Message object
        """
        if not self.is_running:
            return
        
        try:
            # Ignore non-note messages
            if msg.type not in ['note_on', 'note_off']:
                return
            
            # Use message's channel if specified, otherwise use synth's default channel
            channel = msg.channel if hasattr(msg, 'channel') else self.channel
            note_string = self.get_note_name(msg.note)
            
            if msg.type == 'note_on':
                if msg.velocity > 0:
                    # logger.info(f"[SYNTH] Playing note {note_string} ({msg.note} MIDI) with velocity {msg.velocity}")
                    self.synth.noteon(channel, msg.note, msg.velocity)
                else:
                    # Velocity 0 is equivalent to note_off
                    # logger.info(f"[SYNTH] Released note {note_string} ({msg.note} MIDI)")
                    self.synth.noteoff(channel, msg.note)
                    
            elif msg.type == 'note_off':
                # logger.info(f"[SYNTH] Released note {note_string} ({msg.note} MIDI)")
                self.synth.noteoff(channel, msg.note)
                
        except Exception as e:
            logger.error(f"[SYNTH] Error handling MIDI message: {e}")
            logger.error(f"[SYNTH] Message was: {msg}")
    
    
    
    # Closes and cleans up the synthesizer resources
    def cleanup(self):
        if self.synth:
            if self.sfid is not None and self.sfid != -1:
                self.synth.sfunload(self.sfid)
            self.synth.delete()
            logger.info("[SYNTH] Synth closed")
        self.is_running = False
        
    def get_note_name(self, midi_note: int) -> str:
        note_name = NOTES[midi_note % 12]
        octave = (midi_note // 12) - 1
        return f"{note_name}{octave}"