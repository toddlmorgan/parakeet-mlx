"""
Parakeet MLX Web UI - A responsive web interface for the parakeet-mlx ASR project

This script sets up a web-based user interface for the parakeet-mlx ASR (Automatic Speech Recognition)
system, designed specifically for Apple Silicon Macs but compatible with other platforms.

Features:
- Responsive web interface using Gradio
- Support for file uploads (single or batch)
- Audio recording directly from browser
- Visualization of transcription results with word-level timestamps
- Support for local model usage
- Export options for transcription results (SRT, VTT, TXT, JSON)
- Audio file format conversion
"""

import os
import sys
import json
import time
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import gradio as gr
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("parakeet-ui")
console = Console()

# Try to import parakeet_mlx - show informative error if not available
try:
    from parakeet_mlx import from_pretrained, AlignedResult
    from parakeet_mlx.audio import load_audio
    from parakeet_mlx.cli import format_timestamp, to_srt, to_vtt, to_txt, to_json
    PARAKEET_AVAILABLE = True
except ImportError:
    logger.warning("parakeet_mlx not found. Please install with: pip install parakeet-mlx")
    PARAKEET_AVAILABLE = False

# Check for required dependencies for audio conversion
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    logger.warning("ffmpeg-python not found. Installing with: pip install ffmpeg-python")
    FFMPEG_AVAILABLE = False

# Constants
DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"
AVAILABLE_MODELS = [
    "mlx-community/parakeet-tdt-0.6b-v2",
    "mlx-community/parakeet-tdt-0.125b-v2",
    # Add other models as they become available
]
OUTPUT_FORMATS = ["srt", "vtt", "txt", "json"]
DEFAULT_OUTPUT_FORMAT = "srt"
SAMPLE_RATE = 16000  # Hz

class ParakeetModel:
    """Wrapper for Parakeet MLX model interaction"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        self.loaded = False
    
    def load(self) -> bool:
        """Load the model"""
        if not PARAKEET_AVAILABLE:
            logger.error("Cannot load model: parakeet_mlx not installed")
            return False
        
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = from_pretrained(self.model_name)
            self.loaded = True
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def transcribe(
        self, 
        audio_path: Union[str, Path], 
        chunk_duration: Optional[float] = 120.0,
        overlap_duration: float = 15.0,
        progress_callback=None
    ) -> Optional[AlignedResult]:
        """Transcribe audio file"""
        if not self.loaded:
            if not self.load():
                return None
        
        try:
            # Custom progress tracker for UI updates
            def callback(current, total):
                if progress_callback:
                    progress = current / total
                    progress_callback(progress)
            
            # Process transcription
            result = self.model.transcribe(
                audio_path,
                chunk_duration=chunk_duration,
                overlap_duration=overlap_duration,
                chunk_callback=callback
            )
            
            return result
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

class AudioProcessor:
    """Handle audio processing and format conversion"""
    
    @staticmethod
    def convert_audio(input_path: Union[str, Path], output_path: Union[str, Path] = None) -> Optional[Path]:
        """Convert audio to the format required by Parakeet (16kHz WAV)"""
        if not FFMPEG_AVAILABLE:
            logger.error("ffmpeg-python not installed. Cannot convert audio.")
            return None
        
        input_path = Path(input_path)
        
        if output_path is None:
            # Create a temporary file if no output path is provided
            temp_dir = Path(tempfile.gettempdir()) / "parakeet-ui"
            temp_dir.mkdir(exist_ok=True)
            output_path = temp_dir / f"{input_path.stem}_converted.wav"
        else:
            output_path = Path(output_path)
        
        try:
            # Use ffmpeg to convert to 16kHz WAV
            (
                ffmpeg
                .input(str(input_path))
                .output(
                    str(output_path),
                    acodec='pcm_s16le',
                    ac=1,
                    ar=SAMPLE_RATE
                )
                .overwrite_output()
                .run(quiet=True)
            )
            logger.info(f"Converted audio: {input_path} -> {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None
    
    @staticmethod
    def save_recorded_audio(audio_data: Tuple[int, np.ndarray], filename: Union[str, Path] = None) -> Optional[Path]:
        """Save audio recorded from browser to a file"""
        sample_rate, audio_array = audio_data
        
        if filename is None:
            # Create a temporary file if no output path is provided
            temp_dir = Path(tempfile.gettempdir()) / "parakeet-ui"
            temp_dir.mkdir(exist_ok=True)
            filename = temp_dir / f"recording_{int(time.time())}.wav"
        else:
            filename = Path(filename)
        
        try:
            # Save the numpy array to a WAV file
            import scipy.io.wavfile
            scipy.io.wavfile.write(filename, sample_rate, audio_array)
            
            # If the sample rate isn't 16kHz, convert it
            if sample_rate != SAMPLE_RATE:
                return AudioProcessor.convert_audio(filename)
            
            return filename
        except Exception as e:
            logger.error(f"Error saving recorded audio: {e}")
            return None

class ParakeetUI:
    """Gradio UI for Parakeet MLX"""
    
    def __init__(self):
        self.parakeet = ParakeetModel()
        self.audio_processor = AudioProcessor
        self.current_results = {}  # Store results for download
        self.temp_dir = Path(tempfile.gettempdir()) / "parakeet-ui"
        self.temp_dir.mkdir(exist_ok=True)
    
    def load_model(self, model_name: str) -> str:
        """Load a model and return status message"""
        self.parakeet = ParakeetModel(model_name)
        success = self.parakeet.load()
        
        if success:
            return f"✓ Model {model_name} loaded successfully"
        else:
            return f"❌ Failed to load model {model_name}"
    
    def transcribe_file(
        self, 
        audio_file: str, 
        model_name: str, 
        chunk_duration: float,
        highlight_words: bool,
        output_format: str,
        progress=gr.Progress()
    ) -> Tuple[str, str, Dict, str]:
        """Transcribe a single audio file and return results"""
        if not audio_file:
            return "Please upload an audio file.", "", {}, ""
        
        # Ensure model is loaded
        if self.parakeet.model_name != model_name or not self.parakeet.loaded:
            self.load_model(model_name)
        
        # Convert audio if needed
        try:
            converted_path = self.audio_processor.convert_audio(audio_file)
            if not converted_path:
                return "Failed to convert audio file.", "", {}, ""
            
            # Transcribe audio
            result = self.parakeet.transcribe(
                converted_path,
                chunk_duration=chunk_duration if chunk_duration > 0 else None,
                progress_callback=progress
            )
            
            if not result:
                return "Transcription failed.", "", {}, ""
            
            # Generate output based on format
            if output_format == "txt":
                output = to_txt(result)
                file_ext = "txt"
            elif output_format == "vtt":
                output = to_vtt(result, highlight_words=highlight_words)
                file_ext = "vtt"
            elif output_format == "json":
                output = to_json(result)
                file_ext = "json"
            else:  # Default to srt
                output = to_srt(result, highlight_words=highlight_words)
                file_ext = "srt"
            
            # Save output for download
            output_path = self.temp_dir / f"transcript_{int(time.time())}.{file_ext}"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            
            # Create visualization data
            vis_data = self.create_visualization_data(result)
            
            # Store result for potential download
            file_id = Path(audio_file).stem
            self.current_results[file_id] = {
                "result": result,
                "highlight_words": highlight_words,
                "output_path": output_path
            }
            
            return output, to_txt(result), vis_data, str(output_path)
        
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            return f"Error: {str(e)}", "", {}, ""
    
    def transcribe_recording(
        self, 
        recording: Tuple[int, np.ndarray], 
        model_name: str, 
        chunk_duration: float,
        highlight_words: bool,
        output_format: str,
        progress=gr.Progress()
    ) -> Tuple[str, str, Dict, str]:
        """Transcribe recording from browser"""
        if recording is None:
            return "Please record audio first.", "", {}, ""
        
        # Save recording to file
        audio_path = self.audio_processor.save_recorded_audio(recording)
        if not audio_path:
            return "Failed to save recording.", "", {}, ""
        
        # Use the same process as file transcription
        return self.transcribe_file(
            str(audio_path), 
            model_name, 
            chunk_duration, 
            highlight_words,
            output_format,
            progress
        )
    
    def batch_transcribe(
        self,
        files: List[str],
        model_name: str,
        chunk_duration: float,
        highlight_words: bool,
        output_format: str,
        output_dir: str,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """Transcribe multiple files and save results to specified directory"""
        if not files:
            return "Please upload audio files for batch processing.", ""
        
        # Ensure model is loaded
        if self.parakeet.model_name != model_name or not self.parakeet.loaded:
            self.load_model(model_name)
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir) if output_dir else Path.cwd() / "transcripts"
        output_path.mkdir(exist_ok=True)
        
        # Process each file
        results = []
        total_files = len(files)
        
        for i, file in enumerate(files):
            file_path = Path(file)
            progress(i / total_files, f"Processing {file_path.name} ({i+1}/{total_files})")
            
            try:
                # Convert audio
                converted_path = self.audio_processor.convert_audio(file)
                if not converted_path:
                    results.append(f"❌ {file_path.name}: Failed to convert audio")
                    continue
                
                # Transcribe
                result = self.parakeet.transcribe(
                    converted_path,
                    chunk_duration=chunk_duration if chunk_duration > 0 else None,
                    progress_callback=lambda p: progress((i + p) / total_files)
                )
                
                if not result:
                    results.append(f"❌ {file_path.name}: Transcription failed")
                    continue
                
                # Generate output based on format
                if output_format == "txt":
                    output = to_txt(result)
                    file_ext = "txt"
                elif output_format == "vtt":
                    output = to_vtt(result, highlight_words=highlight_words)
                    file_ext = "vtt"
                elif output_format == "json":
                    output = to_json(result)
                    file_ext = "json"
                else:  # Default to srt
                    output = to_srt(result, highlight_words=highlight_words)
                    file_ext = "srt"
                
                # Save output
                output_file = output_path / f"{file_path.stem}.{file_ext}"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output)
                
                results.append(f"✓ {file_path.name}: Saved to {output_file}")
                
            except Exception as e:
                results.append(f"❌ {file_path.name}: Error - {str(e)}")
        
        progress(1.0, "Batch processing complete")
        
        # Generate summary
        summary = f"Processed {total_files} files\n"
        summary += f"Output directory: {output_path}\n\n"
        details = "\n".join(results)
        
        return summary, details
    
    def create_visualization_data(self, result: AlignedResult) -> Dict:
        """Create visualization data for the frontend from transcription result"""
        visualization = {
            "sentences": [],
            "wordLevelData": []
        }
        
        for sentence in result.sentences:
            # Add sentence-level data
            visualization["sentences"].append({
                "text": sentence.text,
                "start": sentence.start,
                "end": sentence.end,
                "duration": sentence.duration
            })
            
            # Add word-level data
            for token in sentence.tokens:
                visualization["wordLevelData"].append({
                    "text": token.text.strip(),
                    "start": token.start,
                    "end": token.end,
                    "duration": token.duration
                })
        
        return visualization
    
    def select_output_directory(self) -> str:
        """Return default directory or let user select one"""
        if sys.platform == "darwin":  # macOS
            return str(Path.home() / "Documents" / "Transcripts")
        else:
            return str(Path.cwd() / "transcripts")
    
    def launch_ui(self, server_name=None, server_port=None):
        """Create and launch the Gradio UI"""
        with gr.Blocks(title="Parakeet MLX - Speech to Text", theme=gr.themes.Soft()) as app:
            gr.Markdown("""
            # 🦜 Parakeet MLX - Speech to Text Interface
            
            A web interface for the [Parakeet MLX](https://github.com/senstella/parakeet-mlx) 
            Automatic Speech Recognition (ASR) system, optimized for Apple Silicon.
            """)
            
            # Settings tab
            with gr.Tab("Settings"):
                gr.Markdown("### Model Configuration")
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=DEFAULT_MODEL,
                        label="Select Model"
                    )
                    load_btn = gr.Button("Load Model", variant="primary")
                    model_status = gr.Textbox(
                        label="Status", 
                        value="Model not loaded. Click 'Load Model' to start.",
                        interactive=False
                    )
                
                gr.Markdown("### Processing Options")
                with gr.Row():
                    chunk_duration = gr.Slider(
                        minimum=0, 
                        maximum=300, 
                        value=120, 
                        step=10,
                        label="Chunk Duration (seconds, 0 to disable chunking)"
                    )
                    highlight_words = gr.Checkbox(
                        label="Highlight Words in Timestamps", 
                        value=False
                    )
                
                gr.Markdown("### Output Options")
                with gr.Row():
                    output_format = gr.Radio(
                        choices=OUTPUT_FORMATS,
                        value=DEFAULT_OUTPUT_FORMAT,
                        label="Output Format"
                    )
            
            # Single file tab
            with gr.Tab("Single File"):
                with gr.Row():
                    with gr.Column():
                        audio_file = gr.File(
                            label="Upload Audio File",
                            file_types=["audio/*"]
                        )
                        transcribe_btn = gr.Button("Transcribe", variant="primary")
                    
                    with gr.Column():
                        audio_recorder = gr.Audio(
                            label="Or Record Audio", 
                            source="microphone",
                            type="numpy"
                        )
                        record_btn = gr.Button("Transcribe Recording", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        transcript_output = gr.TextArea(
                            label="Transcription Output",
                            lines=10,
                            interactive=False
                        )
                        plain_text = gr.TextArea(
                            label="Plain Text",
                            lines=5,
                            interactive=False
                        )
                    
                    with gr.Column():
                        vis_output = gr.JSON(
                            label="Visualization Data",
                            visible=False
                        )
                        download_btn = gr.File(
                            label="Download Transcript",
                            interactive=False,
                            visible=True
                        )
                        with gr.Row():
                            transcript_timeline = gr.HTML(
                                label="Interactive Timeline",
                                value=self.get_timeline_html()
                            )
            
            # Batch processing tab
            with gr.Tab("Batch Processing"):
                with gr.Row():
                    batch_files = gr.File(
                        label="Upload Multiple Audio Files",
                        file_types=["audio/*"],
                        file_count="multiple"
                    )
                
                with gr.Row():
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        value=self.select_output_directory()
                    )
                    output_dir_btn = gr.Button("Browse...", size="sm")
                
                batch_btn = gr.Button("Start Batch Processing", variant="primary")
                
                with gr.Row():
                    batch_summary = gr.Textbox(
                        label="Batch Summary",
                        interactive=False
                    )
                    batch_details = gr.Textbox(
                        label="Processing Details",
                        lines=10,
                        interactive=False
                    )
            
            # About tab
            with gr.Tab("About"):
                gr.Markdown("""
                ### About Parakeet MLX
                
                This application provides a web interface for [Parakeet MLX](https://github.com/senstella/parakeet-mlx),
                an implementation of Nvidia's Parakeet ASR (Automatic Speech Recognition) models for Apple Silicon using MLX.
                
                #### Features
                - Transcribe audio from files or microphone recording
                - Batch process multiple files
                - Support for different output formats (SRT, VTT, TXT, JSON)
                - Visualization of transcription with word-level timestamps
                - Optimized for Apple Silicon Macs
                
                #### Requirements
                - Python 3.10 or newer
                - MLX (Apple's machine learning framework)
                - FFmpeg (for audio conversion)
                
                #### Credits
                - Original Parakeet MLX implementation by [Senstella](https://github.com/senstella)
                - Nvidia for training the original Parakeet models
                - Apple for the MLX framework
                """)
            
            # Connect UI components to handlers
            load_btn.click(
                fn=self.load_model,
                inputs=[model_dropdown],
                outputs=[model_status]
            )
            
            transcribe_btn.click(
                fn=self.transcribe_file,
                inputs=[
                    audio_file, 
                    model_dropdown, 
                    chunk_duration, 
                    highlight_words,
                    output_format
                ],
                outputs=[
                    transcript_output, 
                    plain_text, 
                    vis_output,
                    download_btn
                ]
            )
            
            record_btn.click(
                fn=self.transcribe_recording,
                inputs=[
                    audio_recorder,
                    model_dropdown, 
                    chunk_duration, 
                    highlight_words,
                    output_format
                ],
                outputs=[
                    transcript_output, 
                    plain_text, 
                    vis_output,
                    download_btn
                ]
            )
            
            batch_btn.click(
                fn=self.batch_transcribe,
                inputs=[
                    batch_files,
                    model_dropdown,
                    chunk_duration,
                    highlight_words,
                    output_format,
                    output_dir
                ],
                outputs=[
                    batch_summary,
                    batch_details
                ]
            )
            
            # Load visualization script
            vis_output.change(
                fn=lambda x: self.update_visualization(x),
                inputs=[vis_output],
                outputs=[transcript_timeline]
            )
            
            # Launch the app
            app.launch(
                server_name=server_name if server_name else "127.0.0.1",
                server_port=server_port if server_port else 7860,
                share=False
            )
    
    def get_timeline_html(self) -> str:
        """Generate HTML for the interactive timeline"""
        return """
        <div id="transcript-container" style="width: 100%; height: 300px; overflow-y: auto; padding: 10px; border: 1px solid #ccc; border-radius: 4px;">
            <div id="transcript-timeline" style="position: relative; min-height: 200px;"></div>
        </div>
        
        <script>
        function updateVisualization(data) {
            if (!data || !data.sentences) return;
            
            const container = document.getElementById('transcript-timeline');
            container.innerHTML = '';
            
            // Add timeline ruler
            const totalDuration = data.sentences.length ? 
                Math.max(...data.sentences.map(s => s.end)) : 0;
            
            const ruler = document.createElement('div');
            ruler.className = 'timeline-ruler';
            ruler.style.cssText = 'position: relative; height: 30px; margin-bottom: 15px; border-bottom: 1px solid #888;';
            
            // Add tick marks every 5 seconds
            for (let i = 0; i <= totalDuration; i += 5) {
                const tick = document.createElement('div');
                const percent = (i / totalDuration) * 100;
                tick.style.cssText = `position: absolute; left: ${percent}%; height: 10px; border-left: 1px solid #888;`;
                
                const label = document.createElement('div');
                label.textContent = formatTime(i);
                label.style.cssText = `position: absolute; left: ${percent}%; top: 10px; transform: translateX(-50%); font-size: 12px;`;
                
                ruler.appendChild(tick);
                ruler.appendChild(label);
            }
            
            container.appendChild(ruler);
            
            // Add sentences
            data.sentences.forEach((sentence, idx) => {
                const sentenceEl = document.createElement('div');
                sentenceEl.className = 'sentence';
                sentenceEl.style.cssText = `
                    position: relative; 
                    margin-bottom: 15px;
                    padding: 10px;
                    border-radius: 4px;
                    background-color: #f0f0f0;
                `;
                
                // Add sentence text
                const sentenceText = document.createElement('div');
                sentenceText.textContent = sentence.text;
                sentenceText.style.cssText = 'margin-bottom: 5px; font-weight: bold;';
                
                // Add timestamp
                const timestamp = document.createElement('div');
                timestamp.textContent = `${formatTime(sentence.start)} → ${formatTime(sentence.end)}`;
                timestamp.style.cssText = 'font-size: 12px; color: #666;';
                
                sentenceEl.appendChild(sentenceText);
                sentenceEl.appendChild(timestamp);
                
                // Add word-level visualization if available
                if (data.wordLevelData && data.wordLevelData.length) {
                    // Filter words that belong to this sentence's time range
                    const sentenceWords = data.wordLevelData.filter(
                        word => word.start >= sentence.start && word.end <= sentence.end
                    );
                    
                    if (sentenceWords.length) {
                        const wordsContainer = document.createElement('div');
                        wordsContainer.style.cssText = 'margin-top: 10px; position: relative; height: 30px; background-color: #e0e0e0; border-radius: 4px;';
                        
                        sentenceWords.forEach(word => {
                            if (!word.text.trim()) return;
                            
                            const wordStart = ((word.start - sentence.start) / sentence.duration) * 100;
                            const wordWidth = (word.duration / sentence.duration) * 100;
                            
                            const wordEl = document.createElement('div');
                            wordEl.className = 'word';
                            wordEl.textContent = word.text;
                            wordEl.title = `${word.text} (${formatTime(word.start)} → ${formatTime(word.end)})`;
                            wordEl.style.cssText = `
                                position: absolute;
                                left: ${wordStart}%;
                                width: ${wordWidth}%;
                                height: 30px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                background-color: #3498db;
                                color: white;
                                font-size: 12px;
                                border-radius: 3px;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                white-space: nowrap;
                                cursor: pointer;
                            `;
                            
                            // Add click event to play audio at this timestamp (could be implemented later)
                            wordEl.addEventListener('click', () => {
                                console.log(`Word clicked: ${word.text} at ${word.start}s`);
                                // Future: Implement audio playback at timestamp
                            });
                            
                            wordsContainer.appendChild(wordEl);
                        });
                        
                        sentenceEl.appendChild(wordsContainer);
                    }
                }
                
                container.appendChild(sentenceEl);
            });
        }
        
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            const ms = Math.floor((seconds % 1) * 1000);
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
        }
        </script>
        """
    
    def update_visualization(self, data):
        """Update the visualization with new data"""
        html = self.get_timeline_html()
        script = f"""
        <script>
        (function() {{
            const data = {json.dumps(data)};
            // Call on next tick to ensure DOM is ready
            setTimeout(() => updateVisualization(data), 100);
        }})();
        </script>
        """
        return html + script

def check_dependencies():
    """Check and install required dependencies"""
    missing_deps = []
    
    try:
        import gradio
    except ImportError:
        missing_deps.append("gradio")
    
    try:
        import rich
    except ImportError:
        missing_deps.append("rich")
    
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")
    
    try:
        import ffmpeg
    except ImportError:
        missing_deps.append("ffmpeg-python")
    
    # Install missing dependencies
    if missing_deps:
        print(f"Installing missing dependencies: {', '.join(missing_deps)}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_deps])
        print("Dependencies installed. Restart may be required.")
    
    # Check for parakeet_mlx
    try:
        import parakeet_mlx
    except ImportError:
        print("Parakeet MLX not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "parakeet-mlx"])
            print("Parakeet MLX installed successfully.")
        except Exception as e:
            print(f"Error installing Parakeet MLX: {e}")
            print("Please install manually: pip install parakeet-mlx")

def main():
    """Main entry point"""
    # Check dependencies
    check_dependencies()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Parakeet MLX Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    args = parser.parse_args()
    
    # Launch the UI
    ui = ParakeetUI()
    ui.launch_ui(server_name=args.host, server_port=args.port)

if __name__ == "__main__":
    main()