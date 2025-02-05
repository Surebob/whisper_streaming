#!/usr/bin/env python3
import os
import sys
import time
import logging
import argparse
import threading
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.spinner import Spinner
from rich.progress import BarColumn, SpinnerColumn, Progress, TextColumn, TimeElapsedColumn, MofNCompleteColumn, TimeRemainingColumn
from rich.align import Align
import torch
from rich.columns import Columns
from rich.table import Table
import psutil
import queue
from collections import deque

# Import pipeline and models after other imports
from pipeline import StreamingPipeline
from models.stats import ModelStatsMonitor

# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("transcription.log")
    ]
)

# Disable other loggers from printing to console
logging.getLogger("faster_whisper").setLevel(logging.ERROR)
logging.getLogger("models.diarization").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("pipeline").setLevel(logging.ERROR)

logger = logging.getLogger("pipeline_ui")

# Add at top of file, after imports
MODEL_THRESHOLDS = {
    'vad': (5, 10),
    'whisper': (100, 200),
    'diarizer': (50, 100),
    'pipeline': (150, 300)
}

MODEL_STATS_ORDER = ['vad', 'whisper', 'diarizer', 'pipeline']
MODEL_NAMES = {
    'vad': 'VAD',
    'whisper': 'Whisper',
    'diarizer': 'Diarizer',
    'pipeline': 'Pipeline'
}

def ensure_renderable(content) -> Text:
    """Convert content to Rich renderable if needed."""
    if isinstance(content, (Align, Text)):
        return content
    return Text.from_markup(str(content))

def format_latency_stats(stats: dict) -> list:
    """Format model latency statistics with colors."""
    formatted_stats = []
    for model in MODEL_STATS_ORDER:
        metrics = stats.get(model)
        if metrics:
            avg_latency = metrics['avg'] * 1000
            min_latency = metrics['min'] * 1000
            max_latency = metrics['max'] * 1000
            
            low, high = MODEL_THRESHOLDS[model]
            color = (
                "green" if avg_latency < low else
                "yellow" if avg_latency < high else
                "red"
            )
            
            latency_text = f"[{color}]{avg_latency:.1f}ms ({min_latency:.1f}-{max_latency:.1f})[/{color}]"
            formatted_stats.append((MODEL_NAMES[model], latency_text))
    return formatted_stats

def make_layout(transcript: str, history: str, status: str, vad_status) -> Layout:
    """Creates a layout with transcription, history, status, and VAD panels."""
    layout = Layout()
    
    # Split into main and info sections
    layout.split(
        Layout(name="content", ratio=4),
        Layout(name="info", ratio=1)
    )
    
    # Split content into current and history/assistant sections
    layout["content"].split(
        Layout(name="current", ratio=1),
        Layout(name="middle_section", ratio=2)
    )

    # Split middle section into history and assistant response
    layout["content"]["middle_section"].split_row(
        Layout(name="history", ratio=1),
        Layout(name="assistant", ratio=1)
    )
    
    # Split info section into three columns
    layout["info"].split_row(
        Layout(name="models", ratio=1),
        Layout(name="system", ratio=1),
        Layout(name="vad", ratio=1)
    )
    
    # Create panels with minimal padding for faster rendering
    current_panel = Panel(
        ensure_renderable(transcript),
        title="Live Transcription",
        border_style="blue",
        padding=(0, 1)
    )
    
    history_panel = Panel(
        ensure_renderable(history),
        title="Transcript History",
        border_style="green",
        padding=(0, 1)
    )

    # New assistant response panel
    assistant_panel = Panel(
        Text.from_markup("[grey50]Assistant responses will appear here[/grey50]"),
        title="Assistant Response",
        border_style="magenta",
        padding=(0, 1)
    )
    
    models_panel = Panel(
        status[0] if isinstance(status, tuple) else ensure_renderable("No model info"),
        title="Active Models",
        border_style="magenta",
        padding=(0, 1)
    )
    
    system_panel = Panel(
        status[1] if isinstance(status, tuple) else ensure_renderable("No system info"),
        title="System Status",
        border_style="cyan",
        padding=(0, 1)
    )
    
    vad_panel = Panel(
        vad_status,  # Already a Rich renderable
        title="Voice Activity",
        border_style="yellow",
        padding=(0, 1)
    )
    
    # Update layout sections
    layout["content"]["current"].update(current_panel)
    layout["content"]["middle_section"]["history"].update(history_panel)
    layout["content"]["middle_section"]["assistant"].update(assistant_panel)
    layout["info"]["models"].update(models_panel)
    layout["info"]["system"].update(system_panel)
    layout["info"]["vad"].update(vad_panel)
    
    return layout

def format_transcript(text: str) -> str:
    """Format transcript text with enhanced styling."""
    if not text.strip():
        return Align.center(
            "[grey50]Waiting for speech...\n[dim]Start speaking to begin transcription[/dim][/grey50]",
            vertical="middle"
        )
    
    # Add subtle highlighting for key parts of speech
    highlighted = (
        text.replace(".", "[yellow].[/yellow]")
            .replace("!", "[yellow]![/yellow]")
            .replace("?", "[yellow]?[/yellow]")
    )
    
    return highlighted.strip()

def format_history(pipeline) -> str:
    """Format transcript history with enhanced visuals."""
    if not pipeline.transcript_history:
        return Align.center(
            "[grey50]No transcript history\n[dim]Completed transcripts will appear here[/dim][/grey50]",
            vertical="middle"
        )
    
    # Get last 10 entries
    history = pipeline.transcript_history[-10:]
    formatted_entries = []
    
    for entry in history:
        if not isinstance(entry, dict) or not entry.get('text', '').strip():
            continue
            
        # Enhanced entry formatting with icons and better visual hierarchy
        timestamp = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
        duration = entry['end'] - entry['start']
        duration_color = (
            "green" if duration < 5 else
            "yellow" if duration < 10 else
            "red"
        )
        
        formatted_entries.append(
            f"[grey70]{timestamp}[/grey70] "
            f"[blue]{entry['speaker']}[/blue] "
            f"[{duration_color}]({duration:.1f}s)[/{duration_color}] "
            f"{entry['text'].strip()}"
        )
    
    return "\n".join(formatted_entries)

def get_cpu_usage():
    """Get CPU usage percentage with 1-second caching."""
    try:
        from time import time
        
        # Use class variable to cache stats
        if not hasattr(get_cpu_usage, '_last_update'):
            get_cpu_usage._last_update = 0
            get_cpu_usage._cached_stats = None
            
        # Only update every 1 second
        current_time = time()
        if current_time - get_cpu_usage._last_update < 1.0 and get_cpu_usage._cached_stats is not None:
            return get_cpu_usage._cached_stats
            
        # Get overall CPU usage with a small interval
        cpu_percent = psutil.cpu_percent(interval=0.5)
        
        get_cpu_usage._last_update = current_time
        get_cpu_usage._cached_stats = cpu_percent
        return cpu_percent
            
    except Exception as e:
        logger.warning(f"CPU stats error: {str(e)}")
        return None

def get_gpu_memory():
    """Get GPU memory and utilization info using NVML"""
    try:
        import pynvml
        from time import time
        
        # Use class variable to cache stats
        if not hasattr(get_gpu_memory, '_last_update'):
            get_gpu_memory._last_update = 0
            get_gpu_memory._cached_stats = None
            
        # Only update every 1 second
        current_time = time()
        if current_time - get_gpu_memory._last_update < 1.0 and get_gpu_memory._cached_stats:
            return get_gpu_memory._cached_stats
            
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get total memory info
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = info.total / 1024**3
            used_memory = info.used / 1024**3
            
            # Get utilization rates
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util_rates.gpu
            mem_util = (used_memory / total_memory) * 100
            
            stats = (
                used_memory,      # Used memory in GB
                total_memory,     # Total memory in GB
                gpu_util,        # GPU utilization percentage
                mem_util         # Memory utilization percentage
            )
            
            get_gpu_memory._last_update = current_time
            get_gpu_memory._cached_stats = stats
            return stats
            
        except pynvml.NVMLError as e:
            logger.warning(f"NVML error: {str(e)}")
            return None
            
    except Exception as e:
        logger.warning(f"GPU stats error: {str(e)}")
        return None

class SystemMonitor:
    """Handles system resource monitoring in a separate thread."""
    def __init__(self, update_interval=1.0):
        self.update_interval = update_interval
        self.running = True
        self._thread = None
        self._lock = threading.Lock()
        self._stats = {
            'cpu': None,
            'gpu_memory': None,
            'gpu_util': None
        }
        self._stats_queue = queue.Queue()
        
    def start(self):
        """Start the monitoring thread."""
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            
    def get_stats(self):
        """Get the latest system stats."""
        try:
            # Get all available stats from queue
            while not self._stats_queue.empty():
                stats = self._stats_queue.get_nowait()
                with self._lock:
                    self._stats.update(stats)
        except queue.Empty:
            pass
        
        with self._lock:
            return self._stats.copy()
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        last_update = 0
        
        while self.running:
            current_time = time.time()
            
            # Only update every update_interval seconds
            if current_time - last_update >= self.update_interval:
                stats = {}
                
                # Get CPU usage
                try:
                    stats['cpu'] = psutil.cpu_percent(interval=None)
                except Exception as e:
                    logger.warning(f"CPU stats error: {str(e)}")
                    stats['cpu'] = None
                
                # Get GPU stats if available
                if torch.cuda.is_available():
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        
                        # Get memory info
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_used = info.used / 1024**3
                        memory_total = info.total / 1024**3
                        memory_percent = (memory_used / memory_total) * 100
                        
                        # Get utilization
                        util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util_rates.gpu
                        
                        stats['gpu_memory'] = (memory_used, memory_total, memory_percent)
                        stats['gpu_util'] = gpu_util
                        
                    except Exception as e:
                        logger.warning(f"GPU stats error: {str(e)}")
                        stats['gpu_memory'] = None
                        stats['gpu_util'] = None
                
                # Put stats in queue
                try:
                    self._stats_queue.put(stats)
                except queue.Full:
                    pass
                
                last_update = current_time
            
            time.sleep(0.1)  # Sleep to prevent high CPU usage

def format_status(pipeline, system_monitor) -> tuple:
    """Format status information split into models and system stats."""
    # Models info table
    model_info = Table(show_header=False, box=None, padding=(0, 1),
                      collapse_padding=True, show_edge=False, expand=True)
    
    # Core Models
    model_info.add_row("[bold dim]Core Models[/bold dim]", "")
    model_info.add_row("[dim]Whisper:[/dim]", f"[cyan]{pipeline.model_name}[/cyan]")
    model_info.add_row("[dim]VAD:[/dim]", "[cyan]Silero VAD[/cyan]")
    model_info.add_row("[dim]Diarizer:[/dim]", "[cyan]Pyannote 3.1[/cyan]")
    
    # Speaker Models & Processing
    model_info.add_row("", "")  # Spacer
    model_info.add_row("[bold dim]Speaker Models[/bold dim]", "")
    model_info.add_row("[dim]Speaker:[/dim]", "[cyan]ECAPA-TDNN + X-vector[/cyan]")
    model_info.add_row("[dim]Features:[/dim]", "[cyan]Fbank + Mel Stats[/cyan]")
    model_info.add_row("[dim]Index:[/dim]", "[cyan]FAISS[/cyan]")
    
    # Add inference stats if available
    if hasattr(pipeline, 'model_stats'):
        stats = pipeline.model_stats.get_stats()
        if stats:
            model_info.add_row("", "")  # Spacer
            model_info.add_row("[bold cyan]Inference Stats[/bold cyan]", "")
            
            for model_name, latency_text in format_latency_stats(stats):
                model_info.add_row(f"[dim]{model_name}:[/dim]", latency_text)
    
    # System stats table
    sys_stats = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        collapse_padding=True,
        show_edge=False,
        expand=True
    )
    
    # Get system stats
    stats = system_monitor.get_stats()
    
    # Add CPU usage
    cpu_usage = stats.get('cpu')
    if cpu_usage is not None:
        cpu_color = (
            "green" if cpu_usage < 50 else
            "yellow" if cpu_usage < 80 else
            "red"
        )
        sys_stats.add_row("[dim]CPU:[/dim]", f"[{cpu_color}]{cpu_usage:.1f}%[/{cpu_color}]")
    else:
        sys_stats.add_row("[dim]CPU:[/dim]", "[yellow]N/A[/yellow]")
    
    # Add GPU stats if available
    if torch.cuda.is_available():
        try:
            # Get device properties
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            # Add GPU name
            sys_stats.add_row("[dim]GPU:[/dim]", f"[cyan]{props.name}[/cyan]")
            
            # Add memory stats
            gpu_memory = stats.get('gpu_memory')
            if gpu_memory:
                mem_used, mem_total, mem_util = gpu_memory
                mem_color = (
                    "green" if mem_util < 50 else
                    "yellow" if mem_util < 80 else
                    "red"
                )
                sys_stats.add_row(
                    "[dim]GPU Mem:[/dim]",
                    f"[{mem_color}]{mem_used:.1f}GB[/{mem_color}] / {mem_total:.1f}GB ({mem_util:.1f}%)"
                )
            else:
                sys_stats.add_row("[dim]GPU Mem:[/dim]", "[yellow]N/A[/yellow]")
            
            # Add GPU utilization
            gpu_util = stats.get('gpu_util')
            if gpu_util is not None:
                gpu_color = (
                    "green" if gpu_util < 50 else
                    "yellow" if gpu_util < 80 else
                    "red"
                )
                sys_stats.add_row(
                    "[dim]GPU Load:[/dim]",
                    f"[{gpu_color}]{gpu_util}%[/{gpu_color}]"
                )
            else:
                sys_stats.add_row("[dim]GPU Load:[/dim]", "[yellow]N/A[/yellow]")

            # Add inference stats if available
            if hasattr(pipeline, 'model_stats'):
                stats = pipeline.model_stats.get_stats()
                if stats:
                    sys_stats.add_row("", "")  # Spacer
                    sys_stats.add_row("[bold cyan]Model Latency[/bold cyan]", "")
                    
                    for model_name, latency_text in format_latency_stats(stats):
                        sys_stats.add_row(f"[dim]{model_name}:[/dim]", latency_text)

        except Exception as e:
            logger.warning(f"Error getting GPU stats: {str(e)}")
            sys_stats.add_row("[dim]GPU:[/dim]", "[yellow]CUDA Available[/yellow]")
            sys_stats.add_row("[dim]GPU Mem:[/dim]", "[yellow]N/A[/yellow]")
            sys_stats.add_row("[dim]GPU Load:[/dim]", "[yellow]N/A[/yellow]")
    else:
        sys_stats.add_row("[dim]GPU:[/dim]", "[red]Not Available[/red]")
    
    return model_info, sys_stats

class VADDisplay:
    """Manages the VAD display with progress bars."""
    def __init__(self):
        # Common progress bar settings
        progress_settings = dict(
            expand=True,
            refresh_per_second=60
        )

        # Create progress instances with rich columns
        self.voice_progress = Progress(
            BarColumn(
                bar_width=None,
                style="green",
                complete_style="green",
                finished_style="green",
                pulse_style="green"
            ),
            TextColumn("[magenta]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            **progress_settings
        )
        
        self.buffer_progress = Progress(
            BarColumn(
                bar_width=None,
                style="magenta",
                complete_style="magenta",
                finished_style="magenta",
                pulse_style="magenta"
            ),
            TextColumn("[bright_blue]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            MofNCompleteColumn(),
            **progress_settings
        )
        
        self.silence_progress = Progress(
            BarColumn(
                bar_width=None,
                style="bright_blue",
                complete_style="bright_blue",
                finished_style="bright_blue",
                pulse_style="bright_blue"
            ),
            TextColumn("[bright_yellow]{task.fields[time]:>3.1f}s"),
            TextColumn("•"),
            TimeRemainingColumn(),
            **progress_settings
        )
        
        # Create tasks with descriptions
        self.voice_task = self.voice_progress.add_task("", total=100)
        self.buffer_task = self.buffer_progress.add_task("", total=100)
        self.silence_task = self.silence_progress.add_task("", total=100)
        
        # Create state text
        self.state_text = Text()
        
        # Create progress group with spacing and labels
        self.progress_group = Group(
            self.state_text,
            Text(),  # Empty line for spacing
            Text.from_markup("[cyan]Voice Level"),
            self.voice_progress,
            Text(),  # Empty line for spacing
            Text.from_markup("[bright_blue]Buffer Usage"),
            self.buffer_progress,
            Text(),  # Empty line for spacing
            Text.from_markup("[bright_yellow]Silence"),
            self.silence_progress,
            Text(),  # Empty line for spacing
            Text.from_markup("[dim]Press Ctrl+C to exit[/dim]")
        )
    
    def get_renderable(self, pipeline):
        """Get the renderable group for the VAD display."""
        # Calculate values
        if pipeline.silence_start is None:
            silence_duration = 0
            silence_progress = 0
        else:
            silence_duration = time.time() - pipeline.silence_start
            silence_progress = min(silence_duration / pipeline.silence_threshold, 1.0)

        buffer_size = sum(len(chunk) for chunk in pipeline.audio_buffer)
        buffer_progress = buffer_size / pipeline.max_buffer_size if hasattr(pipeline, 'max_buffer_size') else 0
        voice_level = sum(pipeline.voice_prob_history) / len(pipeline.voice_prob_history) if pipeline.voice_prob_history else 0
        
        # Update progress bars
        self.voice_progress.update(self.voice_task, completed=voice_level * 100)
        self.buffer_progress.update(self.buffer_task, completed=buffer_progress * 100)
        self.silence_progress.update(self.silence_task, completed=silence_progress * 100, time=silence_duration)
        
        # Update state text
        state_text = "• Active" if pipeline.is_speaking else "⏸ Paused"
        state_style = "cyan" if pipeline.is_speaking else "bright_yellow"
        self.state_text.plain = ""  # Clear existing text
        self.state_text.append("State: ", style="cyan")
        self.state_text.append(state_text, style=state_style)
        
        return self.progress_group

def format_vad(pipeline) -> Group:
    """Format VAD information with enhanced visualization."""
    if not hasattr(format_vad, 'display'):
        format_vad.display = VADDisplay()
    return format_vad.display.get_renderable(pipeline)

def main():
    parser = argparse.ArgumentParser(description="Whisper Streaming Transcription UI")
    parser.add_argument("--use-mic", action="store_true", help="Use microphone input")
    parser.add_argument("--model", default="large-v3-turbo", 
                      choices=['tiny.en','tiny','base.en','base','small.en','small',
                              'medium.en','medium','large-v1','large-v2','large-v3','large',
                              'large-v3-turbo','distil-large-v3'])
    parser.add_argument("--compute-type", default="int8_float16", 
                      choices=['int8_float16', 'float16', 'float32', 'int8'])
    parser.add_argument("--min-chunk-size", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Initialize pipeline and monitors
    pipeline = StreamingPipeline(
        model_name=args.model,
        language="en",
        compute_type=args.compute_type,
        min_chunk_size=args.min_chunk_size,
        batch_size=args.batch_size
    )
    
    # Initialize and start system monitor
    system_monitor = SystemMonitor(update_interval=1.0)
    system_monitor.start()
    
    # Create console with optimal settings
    console = Console(color_system="truecolor", force_terminal=True)
    
    # Create VAD display
    vad_display = VADDisplay()
    
    # Initial layout
    layout = make_layout(
        "Initializing...",
        "Loading...",
        ("Starting up...", ""),
        format_vad(pipeline)
    )
    
    # Start microphone processing in background
    mic_thread = threading.Thread(target=pipeline.process_microphone, daemon=True)
    mic_thread.start()
    
    # Main UI loop with maximum refresh rate
    with Live(
        layout,
        console=console,
        screen=True,
        refresh_per_second=60,  # Maximum refresh rate
        transient=True  # Faster updates
    ) as live:
        try:
            while pipeline.running:
                # Update layout with current state
                layout = make_layout(
                    format_transcript(pipeline.live_transcript),
                    format_history(pipeline),
                    format_status(pipeline, system_monitor),
                    format_vad(pipeline)
                )
                live.update(layout)
                
                # Minimal sleep to prevent CPU overload while maintaining responsiveness
                time.sleep(0.001)  # 1ms delay
                
        except KeyboardInterrupt:
            pipeline.running = False
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Error in UI loop: {str(e)}", exc_info=True)
        finally:
            system_monitor.stop()
            pipeline.cleanup()

if __name__ == "__main__":
    main() 