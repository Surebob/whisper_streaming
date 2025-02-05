from typing import Dict, List, Optional, Tuple
from colorama import Fore, Back, Style
import time

class BufferVisualizer:
    """Handles visualization of audio buffer and speech segments."""
    
    def __init__(self, max_duration: float, width: int = 80):
        """Initialize the buffer visualizer.
        
        Args:
            max_duration: Maximum duration in seconds to display
            width: Width of the visualization in characters
        """
        self.max_duration = max_duration
        self.width = width
        self.zoom_factor = 3.0  # How much to zoom the silence window
        
        # Speaker colors and block characters for visualization
        self.speaker_styles = {
            'SPEAKER_00': (Fore.GREEN, "Speaker A", '█'),
            'SPEAKER_01': (Fore.YELLOW, "Speaker B", '▓'),
            'SPEAKER_02': (Fore.BLUE, "Speaker C", '▒'),
            'SPEAKER_03': (Fore.MAGENTA, "Speaker D", '░'),
        }
    
    def _get_position_in_bar(self, time_point: float, window_start: float) -> int:
        """Convert a time point to a position in the visualization bar."""
        relative_pos = time_point - window_start
        return int((relative_pos / self.max_duration) * self.width)
    
    def _get_silence_color(self, progress: float) -> str:
        """Get color for silence visualization based on progress (0.0 to 1.0)."""
        if progress < 0.5:
            return Fore.YELLOW
        elif progress < 0.75:
            return Fore.RED + Style.DIM
        else:
            return Fore.RED + Style.BRIGHT
    
    def draw_buffer(self,
                   current_duration: float,
                   window_start: float,
                   speech_segments: List,
                   pauses: List[Tuple[float, float]],
                   current_pause: Optional[Tuple[float, float]] = None,
                   speech_prob: Optional[float] = None,
                   is_speaking: bool = False,
                   active_segment: Optional[object] = None) -> str:
        """Draw a visualization of the current buffer state.
        
        Args:
            current_duration: Current duration of the buffer in seconds.
            window_start: Start time of the current window.
            speech_segments: List of completed speech segments.
            pauses: List of pause tuples.
            current_pause: Current pause tuple if any.
            speech_prob: Speech probability value if applicable.
            is_speaking: Flag indicating if speech is active.
            active_segment: The active speech segment (with no end_time) if available.
        """
        # Calculate window end and current position
        window_end = window_start + self.max_duration
        current_pos = int((current_duration * self.width) / self.max_duration)
        
        # Initialize the progress bar - always use dim blue dots for silence
        bar = [Fore.BLUE + Style.DIM + '░' + Style.RESET_ALL] * self.width
        
        # First handle active speech and speech probability
        if speech_prob is not None and speech_prob > 0.5:
            # Show speech probability as intensity of green for the current position
            intensity = min(int(speech_prob * 4), 3)  # 0-3 levels of intensity
            symbols = ['░', '▒', '▓', '█']
            bar[current_pos] = Fore.GREEN + symbols[intensity] + Style.RESET_ALL
            
            # If we're actively speaking, fill the segment from start
            if is_speaking and active_segment is not None:
                speech_start_pos = max(0, int((active_segment.start_time - window_start) * self.width / self.max_duration))
                # Fill from start to current with solid blocks
                for i in range(speech_start_pos, current_pos):
                    bar[i] = Fore.GREEN + '█' + Style.RESET_ALL
        
        # Then overlay completed speech segments
        for segment in speech_segments:
            if segment.end_time is not None and segment.end_time >= window_start and segment.start_time <= window_end:
                # Calculate positions in the bar
                start_pos = max(0, int((segment.start_time - window_start) * self.width / self.max_duration))
                end_pos = min(self.width, int((segment.end_time - window_start) * self.width / self.max_duration))
                
                # Fill segment
                for i in range(start_pos, end_pos):
                    bar[i] = Fore.GREEN + '█' + Style.RESET_ALL
        
        # If we're in a pause, show the progress towards silence threshold
        if current_pause:
            pause_start, remaining = current_pause
            if window_start <= pause_start <= window_end:
                silence_window = 2.0  # 2 seconds total
                elapsed = silence_window - remaining
                progress = elapsed / silence_window
                
                # Calculate zoomed window size
                base_width = int(silence_window * self.width / self.max_duration)
                zoomed_width = int(base_width * self.zoom_factor)
                
                # Calculate pause position with zoom
                pause_pos = int((pause_start - window_start) * self.width / self.max_duration)
                if 0 <= pause_pos < self.width:
                    # Calculate how much space we have after the pause
                    space_after = self.width - pause_pos
                    actual_width = min(zoomed_width, space_after)
                    
                    # Calculate proportions for elapsed and remaining
                    elapsed_width = int(actual_width * progress)
                    
                    # Show elapsed silence with color progression
                    color = self._get_silence_color(progress)
                    for i in range(pause_pos, min(pause_pos + elapsed_width, self.width)):
                        bar[i] = color + '▓' + Style.RESET_ALL
                    
                    # Show remaining needed
                    for i in range(pause_pos + elapsed_width, min(pause_pos + actual_width, self.width)):
                        bar[i] = Fore.YELLOW + Style.DIM + '░' + Style.RESET_ALL
                    
                    # Add pause marker
                    if 0 <= pause_pos < self.width:
                        bar[pause_pos] = Fore.YELLOW + '◆' + Style.RESET_ALL
        
        # Add cursor last
        if 0 <= current_pos < self.width:
            if is_speaking:
                bar[current_pos] = Back.GREEN + '►' + Style.RESET_ALL
            else:
                bar[current_pos] = Back.WHITE + Fore.BLACK + '►' + Style.RESET_ALL
        
        # Build status line
        status = f"Buffer: {current_duration:.1f}s/{self.max_duration:.1f}s "
        status += f"[{''.join(bar)}] "
        
        if current_pause:
            _, remaining = current_pause
            color = self._get_silence_color(1 - remaining/2.0)
            status += f"{color}Pause: {remaining:.1f}s{Style.RESET_ALL}"
        elif speech_prob is not None:
            status += f"{Fore.GREEN}Speech: {speech_prob:.2f}{Style.RESET_ALL}"
        else:
            status += f"{Fore.WHITE}Listening...{Style.RESET_ALL}"
        
        return f"\r{status}"
    
    def draw_timeline(self, duration: float, segments: List[dict]) -> str:
        """Draw a timeline visualization of speech segments."""
        # Create the main timeline
        timeline = [Fore.WHITE + '·' + Style.RESET_ALL] * self.width
        
        # Track which speakers are present
        active_speakers = set()
        
        # Mark speech segments
        for segment in segments:
            start_pos = int((segment['start'] / duration) * self.width)
            end_pos = int((segment['end'] / duration) * self.width)
            speaker_info = self.speaker_styles.get(segment['speaker'], (Fore.WHITE, "Unknown", '█'))
            color = speaker_info[0]
            block_char = speaker_info[2]
            active_speakers.add(segment['speaker'])
            
            for i in range(start_pos, min(end_pos + 1, self.width)):
                timeline[i] = color + block_char + Style.RESET_ALL
        
        # Format duration as MM:SS
        duration_formatted = f"{int(duration // 60)}:{int(duration % 60):02d}"
        
        # Create speaker legend
        legend = "\nSpeakers detected:"
        for speaker in sorted(active_speakers):
            color, name, block_char = self.speaker_styles.get(speaker, (Fore.WHITE, "Unknown", '█'))
            legend += f"\n{color}{block_char}{Style.RESET_ALL} {name} ({speaker})"
        
        # Build the complete timeline display
        timeline_str = (
            f"\nTimeline: [{''.join(timeline)}]"
            f"\nDuration: {duration_formatted}"
            f"{legend}"
        )
        
        return timeline_str 