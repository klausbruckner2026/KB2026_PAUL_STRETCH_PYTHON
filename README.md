# Modern Paulstretch

A modernized Python implementation of Paul Nasca's Paulstretch algorithm for extreme audio time-stretching.

## Features

- **Extreme stretching** - Stretch audio by factors of 10x, 100x or more
- **High quality** - Maintains spectral quality through FFT processing
- **Stereo support** - Processes stereo audio correctly
- **Onset protection** - Optional transient preservation for percussive sounds
- **Simple API** - Easy to use in your own projects
- **Command-line interface** - Process files directly from terminal

## Installation

```bash
pip install numpy scipy soundfile

# Basic stretching
python paulstretch.py input.wav output.wav -s 8.0

# With onset protection for drums
python paulstretch.py drums.wav drums_stretched.wav -s 20.0 -o

# Custom window size
python paulstretch.py ambient.wav ambient_stretched.wav -s 16.0 -w 0.5
```