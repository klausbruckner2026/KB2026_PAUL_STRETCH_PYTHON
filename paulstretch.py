"""
Paulstretch - Python Implementation
====================================
A modernized version of Paul Nasca's Paulstretch algorithm for extreme audio time-stretching.

Original algorithm by Nasca Octavian PAUL (Public Domain)
http://www.paulnasca.com/
"""

import numpy as np
from scipy import signal, fftpack
import warnings
from typing import Optional, Tuple, Union
import os


class PaulStretch:
    """
    Paulstretch audio time-stretching algorithm implementation.
    
    This algorithm can stretch audio by extreme factors while maintaining
    spectral quality through windowed FFT processing and phase randomization.
    
    Parameters
    ----------
    stretch_factor : float, default=8.0
        Time stretch factor (>1.0 stretches, <1.0 compresses)
    window_size : float, default=0.25
        Window size in seconds
    onset_protection : bool, default=False
        Enable onset detection for better transient preservation
    random_phase : bool, default=True
        Randomize phases for smoother stretching
    """
    
    def __init__(self, 
                 stretch_factor: float = 8.0,
                 window_size: float = 0.25,
                 onset_protection: bool = False,
                 random_phase: bool = True):
        
        self.stretch_factor = stretch_factor
        self.window_size = window_size
        self.onset_protection = onset_protection
        self.random_phase = random_phase
        
        # Internal state
        self.sample_rate = None
        self.window_samples = None
        self.hop_size = None
        self.fft_size = None
        self.window = None
        
    def _initialize_params(self, sample_rate: int):
        """Initialize processing parameters based on sample rate."""
        self.sample_rate = sample_rate
        self.window_samples = int(self.window_size * sample_rate)
        
        # Make window size even and power of 2 for FFT efficiency
        self.window_samples = self.window_samples // 2 * 2
        self.window_samples = max(4, self.window_samples)
        
        self.fft_size = self.window_samples
        self.hop_size = self.window_samples // 4
        
        # Create analysis window (Hann window for good spectral properties)
        self.window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.window_samples) / self.window_samples))
        
        # Normalize window for perfect reconstruction
        self.window = self.window / np.sqrt(np.sum(self.window**2))
        
    def _process_frame(self, 
                       frame: np.ndarray, 
                       phase_buffer: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single audio frame.
        
        Returns
        -------
        processed_frame : np.ndarray
            The processed time-domain frame
        new_phase : np.ndarray
            Phase information for the next frame
        """
        # Apply window
        windowed = frame * self.window
        
        # FFT
        spectrum = fftpack.fft(windowed)
        
        # Get magnitude and phase
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        if self.random_phase:
            # Randomize phase for smoother stretching
            if phase_buffer is None:
                # Generate new random phase
                new_phase = np.random.uniform(-np.pi, np.pi, len(phase))
            else:
                # Use previous phase with small random variations
                phase_noise = np.random.uniform(-0.1, 0.1, len(phase))
                new_phase = phase_buffer + phase_noise
        else:
            # Maintain original phase progression
            if phase_buffer is not None:
                phase_diff = phase - phase_buffer
                phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                new_phase = phase_buffer + phase_diff
            else:
                new_phase = phase
        
        # Reconstruct spectrum with new phase
        new_spectrum = magnitude * np.exp(1j * new_phase)
        
        # Inverse FFT
        processed = np.real(fftpack.ifft(new_spectrum))
        
        # Re-apply window
        processed = processed * self.window
        
        return processed, new_phase
    
    def stretch(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply Paulstretch to the input audio.
        
        Parameters
        ----------
        audio : np.ndarray
            Input audio signal (mono or stereo)
        sample_rate : int
            Audio sample rate in Hz
            
        Returns
        -------
        stretched : np.ndarray
            Time-stretched audio signal
        """
        # Initialize parameters
        self._initialize_params(sample_rate)
        
        # Handle stereo audio
        if audio.ndim == 2:
            # Process each channel separately
            stretched_channels = []
            for channel in range(audio.shape[1]):
                stretched = self._stretch_mono(audio[:, channel])
                stretched_channels.append(stretched)
            
            # Interleave channels
            max_len = max(len(ch) for ch in stretched_channels)
            stretched = np.zeros((max_len, len(stretched_channels)))
            for i, ch in enumerate(stretched_channels):
                stretched[:len(ch), i] = ch
            return stretched
        else:
            # Mono audio
            return self._stretch_mono(audio)
    
    def _stretch_mono(self, audio: np.ndarray) -> np.ndarray:
        """Apply Paulstretch to mono audio."""
        # Ensure audio is float
        audio = audio.astype(np.float32)
        
        # Calculate output length
        output_length = int(len(audio) * self.stretch_factor)
        
        # Initialize output buffer
        stretched = np.zeros(output_length + self.window_samples)
        
        # Phase buffers for each channel
        phase_buffer = None
        
        # Process frames
        input_position = 0
        output_position = 0
        
        window_samples = self.window_samples
        hop_size = self.hop_size
        stretch_factor = self.stretch_factor
        
        # Pre-calculate indices for efficiency
        output_step = int(hop_size * stretch_factor)
        
        while input_position + window_samples < len(audio):
            # Extract input frame
            frame = audio[input_position:input_position + window_samples]
            
            # Process frame
            processed_frame, phase_buffer = self._process_frame(frame, phase_buffer)
            
            # Add to output with overlap-add
            output_end = min(output_position + window_samples, len(stretched))
            stretched[output_position:output_end] += processed_frame[:output_end - output_position]
            
            # Update positions
            input_position += hop_size
            output_position += output_step
        
        # Trim output
        stretched = stretched[:output_length]
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(stretched))
        if max_val > 0:
            stretched = stretched / max_val * 0.95
        
        return stretched


class PaulStretchWithOnset(PaulStretch):
    """
    Extended Paulstretch algorithm with onset detection for better transient preservation.
    
    This version detects transients in the audio and uses shorter windows
    around onsets to preserve attack characteristics.
    """
    
    def __init__(self,
                 stretch_factor: float = 8.0,
                 window_size: float = 0.25,
                 onset_window_size: float = 0.05,
                 onset_threshold: float = 2.0,
                 random_phase: bool = True):
        
        super().__init__(stretch_factor, window_size, True, random_phase)
        self.onset_window_size = onset_window_size
        self.onset_threshold = onset_threshold
        
    def _detect_onsets(self, audio: np.ndarray) -> np.ndarray:
        """Simple onset detection based on energy envelope derivative."""
        # Compute energy envelope
        frame_size = int(0.01 * self.sample_rate)  # 10ms frames
        hop_size = frame_size // 4
        
        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy.append(np.sqrt(np.mean(frame**2)))
        
        energy = np.array(energy)
        
        # Find onsets where energy derivative exceeds threshold
        energy_diff = np.diff(energy)
        energy_diff = np.concatenate(([0], energy_diff))
        
        # Normalize
        if np.std(energy_diff) > 0:
            energy_diff = energy_diff / np.std(energy_diff)
        
        # Find onset positions
        onset_positions = np.where(energy_diff > self.onset_threshold)[0] * hop_size
        
        return onset_positions
    
    def _stretch_mono(self, audio: np.ndarray) -> np.ndarray:
        """Apply Paulstretch with onset protection."""
        # Detect onsets
        onset_positions = self._detect_onsets(audio)
        
        # Create onset map for adaptive processing
        is_onset = np.zeros(len(audio), dtype=bool)
        onset_radius = int(0.05 * self.sample_rate)  # 50ms radius around onsets
        
        for pos in onset_positions:
            start = max(0, pos - onset_radius)
            end = min(len(audio), pos + onset_radius)
            is_onset[start:end] = True
        
        # Process with adaptive window sizes
        output_length = int(len(audio) * self.stretch_factor)
        stretched = np.zeros(output_length + self.window_samples)
        
        phase_buffer = None
        input_position = 0
        output_position = 0
        
        while input_position + self.window_samples < len(audio):
            # Check if current frame contains an onset
            frame_end = input_position + self.window_samples
            frame_contains_onset = np.any(is_onset[input_position:frame_end])
            
            if frame_contains_onset:
                # Use smaller window for onset regions
                current_window = int(self.onset_window_size * self.sample_rate)
                current_window = current_window // 2 * 2
                current_window = min(current_window, self.window_samples)
            else:
                current_window = self.window_samples
            
            # Extract and process frame
            frame = audio[input_position:input_position + current_window]
            
            # Apply appropriate window
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(current_window) / current_window))
            window = window / np.sqrt(np.sum(window**2))
            windowed = frame * window
            
            # Process frame
            spectrum = fftpack.fft(windowed)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            if self.random_phase:
                if phase_buffer is None:
                    new_phase = np.random.uniform(-np.pi, np.pi, len(phase))
                else:
                    # Use less phase randomization for onsets to preserve transients
                    if frame_contains_onset:
                        phase_noise = np.random.uniform(-0.02, 0.02, len(phase))
                    else:
                        phase_noise = np.random.uniform(-0.1, 0.1, len(phase))
                    new_phase = phase_buffer[:len(phase)] + phase_noise
            else:
                if phase_buffer is not None:
                    phase_diff = phase - phase_buffer[:len(phase)]
                    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                    new_phase = phase_buffer[:len(phase)] + phase_diff
                else:
                    new_phase = phase
            
            # Reconstruct
            new_spectrum = magnitude * np.exp(1j * new_phase)
            processed = np.real(fftpack.ifft(new_spectrum)) * window
            phase_buffer = new_phase
            
            # Overlap-add
            output_end = min(output_position + current_window, len(stretched))
            stretched[output_position:output_end] += processed[:output_end - output_position]
            
            # Update positions
            input_position += self.hop_size
            output_position += int(self.hop_size * self.stretch_factor)
        
        stretched = stretched[:output_length]
        
        # Normalize
        max_val = np.max(np.abs(stretched))
        if max_val > 0:
            stretched = stretched / max_val * 0.95
        
        return stretched


# Simple command-line interface
def main():
    """Simple CLI for Paulstretch."""
    import argparse
    import soundfile as sf
    
    parser = argparse.ArgumentParser(description="Paulstretch - Extreme audio time stretching")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("-s", "--stretch", type=float, default=8.0, help="Stretch factor (default: 8.0)")
    parser.add_argument("-w", "--window", type=float, default=0.25, help="Window size in seconds (default: 0.25)")
    parser.add_argument("-o", "--onset", action="store_true", help="Enable onset protection")
    parser.add_argument("-n", "--no-phase-random", dest="random_phase", action="store_false", 
                       help="Disable phase randomization")
    
    args = parser.parse_args()
    
    # Read input file
    print(f"Loading {args.input}...")
    audio, sr = sf.read(args.input)
    
    # Convert to float if needed
    if audio.dtype != np.float32 and audio.dtype != np.float64:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    
    # Create stretcher
    if args.onset:
        stretcher = PaulStretchWithOnset(
            stretch_factor=args.stretch,
            window_size=args.window,
            random_phase=args.random_phase
        )
    else:
        stretcher = PaulStretch(
            stretch_factor=args.stretch,
            window_size=args.window,
            random_phase=args.random_phase
        )
    
    # Process
    print(f"Stretching by factor {args.stretch}...")
    stretched = stretcher.stretch(audio, sr)
    
    # Save output
    print(f"Saving to {args.output}...")
    sf.write(args.output, stretched, sr)
    
    print("Done!")


if __name__ == "__main__":
    main()