# =============================================================================
# Main class for representing chords and generating MIDI messages
# =============================================================================

from mido import Message
from typing import List
from src.config import NOTE_TO_MIDI_MAP, INTERVALS_MAP

class Chord:
    def __init__(self, root: str, chord_type: str = 'major', bpm: int = 120, beats_per_bar: float = 4.0, velocity: int = 80, channel: int = 0):
        """
        Chord class
        
        Args:
            root: Root note ('C', 'F#', Bb, ...)
            chord_type: Chord type ('major', 'minor', '7', etc.)
            bpm: Beats per minute
            beats_per_bar: Duration in beats (4.0 = one bar in 4/4)
            velocity: MIDI velocity (0-127)
            channel: MIDI channel (0-15)
        """
        self.root = root
        self.chord_type = chord_type
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.velocity = velocity
        self.channel = channel
        
        # Timing calculations
        self.beat_duration = 60.0 / self.bpm
        self.duration_seconds = self.beats_per_bar * self.beat_duration
        
        # Generate MIDI notes
        self.midi_notes = self._generate_midi_notes()
        # Pre-compute MIDI messages for performance optimization
        self.midi_messages = self._generate_midi_messages()
      
        
    # Generates list of MIDI notes for the chord
    def _generate_midi_notes(self, octave: int = 4) -> List[int]:
        # MIDI value for root
        root_midi = octave * 12 + NOTE_TO_MIDI_MAP[self.root]
        # Get intervals by chord type, default = major
        intervals = INTERVALS_MAP.get(self.chord_type, INTERVALS_MAP['major'])
        
        return [root_midi + interval for interval in intervals]

    # Pre-generates MIDI messages for performance optimization using relative timing (start_time=0)
    def _generate_midi_messages(self) -> List[tuple]:
        
        messages = []
        
        for note in self.midi_notes:
            # Note ON message at start
            messages.append((0.0, Message('note_on', channel=self.channel, note=note, velocity=self.velocity)))
            
            # Note OFF message at end of duration
            messages.append((self.duration_seconds, Message('note_off', channel=self.channel, note=note, velocity=0)))
            
        return sorted(messages, key=lambda x: x[0])  # Sort by time
    
    # Update BPM and thus beats and bars - regenerate messages if timing changes
    def update_timing(self, new_bpm: int):
        self.bpm = new_bpm
        self.beat_duration = 60.0 / new_bpm
        self.duration_seconds = self.beats_per_bar * self.beat_duration
        # Regenerate MIDI events with new timing
        self.midi_messages = self._generate_midi_messages()
    
    def __str__(self):
        return f"{self.root}{self.chord_type} ({self.beats_per_bar} beats @ {self.bpm} BPM)"
    
    
# Testing audio playback   
if __name__ == '__main__':
    from src.audio.midi_io import play_chord_sequence
    
    bpm = 80
    progression = [
        Chord('A', bpm=bpm), 
        Chord('F#', 'minor', bpm=bpm), 
        Chord('B', 'minor', bpm=bpm), 
        Chord('E', '7', bpm=bpm),
        Chord('C', 'major', bpm=bpm),
        Chord('C', 'maj7', bpm=bpm),
        Chord('C', '7', bpm=bpm)
    ]
    print("Testing chord progression")
    play_chord_sequence(progression, 'IAC Piano IN')