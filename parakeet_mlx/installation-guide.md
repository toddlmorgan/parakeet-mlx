"""
Installation and User Guide for Parakeet MLX Web UI

This guide provides instructions for installing and using the Parakeet MLX Web UI 
for speech-to-text transcription on macOS with Apple Silicon.
"""

# Parakeet MLX Web UI: Installation & User Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [One-Step Installation](#one-step-installation)
3. [Getting Started](#getting-started)
4. [Using the Interface](#using-the-interface)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [Additional Resources](#additional-resources)

## System Requirements

- **macOS** with Apple Silicon (M1, M2, M3, or later)
- **Python 3.10** or newer
- **FFmpeg** (for audio conversion)
- At least **4GB** of available RAM
- At least **2GB** of disk space for models and temporary files

## One-Step Installation

The Parakeet MLX Web UI uses `uv` for streamlined dependency management. Installation is a single-step process:

```bash
# Create a directory for the app (optional)
mkdir parakeet-ui
cd parakeet-ui

# Download the app script
curl -o parakeet_ui.py https://raw.githubusercontent.com/yourusername/parakeet-mlx-webui/main/parakeet_ui.py

# Run the app (it will auto-install all dependencies)
python parakeet_ui.py
```

That's it! The script will automatically:
1. Install `uv` if not present
2. Use `uv` to install all required dependencies
3. Install Parakeet MLX if not present
4. Check for FFmpeg and provide installation instructions if needed
5. Launch the web interface

### Installing FFmpeg

FFmpeg is required for audio conversion. If not already installed:

```bash
# Using Homebrew (recommended)
brew install ffmpeg

# OR using MacPorts
port install ffmpeg
```

## Getting Started

### First-Time Setup

When you first open the web UI:

1. Go to the **Settings** tab
2. Click **Load Model** to download and initialize the default model
3. Wait for the "Model loaded successfully" message

The model will be downloaded automatically the first time you use it.

## Using the Interface

The interface is organized into several tabs:

### Settings Tab

- **Model Selection**: Choose from available Parakeet MLX models
- **Chunk Duration**: Set the length of audio chunks for processing (0 to disable chunking)
- **Highlight Words**: Enable word-level timestamps in subtitles
- **Output Format**: Choose between SRT, VTT, TXT, and JSON formats

### Single File Tab

For processing individual audio files:

1. Upload an audio file or record directly from your microphone
2. Click **Transcribe** or **Transcribe Recording**
3. View results in the output area and interactive timeline
4. Download the transcript using the download button

### Batch Processing Tab

For processing multiple audio files at once:

1. Upload multiple audio files
2. Set your desired output directory
3. Click **Start Batch Processing**
4. Monitor progress and review results summary

## Advanced Features

### Interactive Timeline

The interactive timeline visualization shows:
- Sentence-level timestamps
- Word-level timestamps (when available)
- Time ruler for navigation

### Format Options

- **SRT**: Standard subtitle format with timestamps
- **VTT**: Web Video Text Tracks format for web videos
- **TXT**: Plain text transcription without timestamps
- **JSON**: Complete data including all timestamp information

### Command Line Options

You can customize the server address and port:

```bash
python parakeet_ui.py --host 0.0.0.0 --port 8080
```

## Troubleshooting

### Common Issues

**Model fails to load**
- Ensure you have a stable internet connection for the initial model download
- Check that you have enough disk space and RAM available
- Try restarting the application

**Audio conversion fails**
- Verify that FFmpeg is properly installed: `ffmpeg -version`
- Try converting the audio manually with FFmpeg before uploading

**Performance issues**
- Try using a smaller model (e.g., parakeet-tdt-0.125b-v2 instead of the default)
- Increase chunk duration for better performance on longer files
- Close other resource-intensive applications

### Dependency Issues

If you encounter issues with dependencies:

```bash
# Manual installation of dependencies
python -m pip install uv
python -m uv pip install parakeet-mlx gradio rich scipy ffmpeg-python
```

## Additional Resources

- [Parakeet MLX GitHub Repository](https://github.com/senstella/parakeet-mlx)
- [MLX Framework Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [Gradio Documentation](https://www.gradio.app/docs)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [UV Documentation](https://github.com/astral-sh/uv)

## License

Parakeet MLX is available under the Apache 2.0 license. See the LICENSE file in the GitHub repository for details.