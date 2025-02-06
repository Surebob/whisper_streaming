import pyaudiowpatch as pyaudio
import wave
import time
import numpy as np
import sys
import os

def list_audio_devices():
    """List all available audio devices."""
    p = pyaudio.PyAudio()
    
    # Get WASAPI info
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        print("\nWASAPI Audio Devices:")
        print("-" * 80)
        
        # List all devices
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info.get('hostApi') == wasapi_info.get('index'):
                print(f"[{i}] {dev_info['name']}")
                print(f"    Max Input Channels: {dev_info['maxInputChannels']}")
                print(f"    Max Output Channels: {dev_info['maxOutputChannels']}")
                print(f"    Default Sample Rate: {dev_info['defaultSampleRate']}")
                print(f"    Is Loopback: {dev_info.get('isLoopbackDevice', False)}")
                print("-" * 80)
    except OSError:
        print("WASAPI is not available on this system")
    finally:
        p.terminate()

def record_system_audio(duration=10, output_file="test_recording.wav"):
    """Record system audio for specified duration."""
    p = pyaudio.PyAudio()
    
    try:
        # Get WASAPI info
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        print("\nAvailable Host APIs:")
        for i in range(p.get_host_api_count()):
            api_info = p.get_host_api_info_by_index(i)
            print(f"[{i}] {api_info['name']}")
        
        # Get default speakers
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        print(f"\nDefault Output Device: {default_speakers['name']}")
        
        # Find loopback device
        loopback_device = None
        for device_index in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(device_index)
            if device_info.get('isLoopbackDevice', False):
                print(f"Found loopback device: {device_info['name']}")
                loopback_device = device_info
                break
        
        if not loopback_device:
            raise Exception("No loopback device found")
        
        print(f"\nRecording from: {loopback_device['name']}")
        print(f"Sample Rate: {int(loopback_device['defaultSampleRate'])}Hz")
        print(f"Channels: {loopback_device['maxInputChannels']}")
        
        # Create audio buffer with frame counter
        frames = []
        frame_count = 0
        
        def callback(in_data, frame_count, time_info, status):
            """Store audio data and show levels."""
            nonlocal frames
            
            if status:
                print(f"Status: {status}")
            
            # Convert to numpy array to check levels
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            max_level = np.max(np.abs(audio_data))
            
            # Only store if we have actual audio data
            if max_level > 0:
                frames.append(in_data)
                sys.stdout.write(f"\rAudio Level: {max_level:10d} | Frames: {len(frames)}")
                sys.stdout.flush()
            
            return (in_data, pyaudio.paContinue)
        
        # Create and start the stream
        chunk_size = 1024 * 4  # Even larger chunk size
        stream = p.open(
            format=pyaudio.paInt16,
            channels=loopback_device['maxInputChannels'],
            rate=int(loopback_device['defaultSampleRate']),
            frames_per_buffer=chunk_size,
            input=True,
            input_device_index=loopback_device['index'],
            stream_callback=callback
        )
        
        print(f"\nRecording for {duration} seconds...")
        print("Please play some audio...")
        
        stream.start_stream()
        
        # Wait and show progress
        for i in range(duration):
            time.sleep(1)
            sys.stdout.write(f"\rRecording: {i+1}/{duration}s | Frames captured: {len(frames)}")
            sys.stdout.flush()
        
        print("\nStopping stream...")
        stream.stop_stream()
        stream.close()
        
        if len(frames) == 0:
            print("\nNo audio data was captured!")
            return
        
        print(f"\nCaptured {len(frames)} frames of audio data")
        
        # Write audio data to file
        print(f"Writing to {output_file}...")
        wave_file = wave.open(output_file, 'wb')
        wave_file.setnchannels(loopback_device['maxInputChannels'])
        wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(int(loopback_device['defaultSampleRate']))
        wave_file.writeframes(b''.join(frames))
        wave_file.close()
        
        # Verify the file
        with wave.open(output_file, 'rb') as wf:
            print(f"\nVerifying recorded file:")
            print(f"Channels: {wf.getnchannels()}")
            print(f"Sample width: {wf.getsampwidth()}")
            print(f"Frame rate: {wf.getframerate()}")
            print(f"Frame count: {wf.getnframes()}")
            print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        p.terminate()

if __name__ == "__main__":
    # First list all devices
    list_audio_devices()
    
    # Then record system audio
    record_system_audio(duration=10, output_file="system_audio_test.wav") 