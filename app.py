import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
from flask import Flask, render_template, request, jsonify
import base64
import tempfile
from gtts import gTTS
from datetime import datetime
import requests
import shutil
import time
import threading
import multiprocessing
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import voice_service as vs
from rag.AIVoiceAssistant import AIVoiceAssistant

# Use smaller model size for faster performance
DEFAULT_MODEL_SIZE = "small"
DEFAULT_CHUNK_LENGTH = 5  # Reduced chunk length for faster processing

# Initialize the AI Hotel Receptionist in a background thread
ai_assistant = None
whisper_model = None

def initialize_models():
    global ai_assistant, whisper_model
    print("Initializing AI models...")
    start_time = time.time()
    
    # Initialize AI assistant
    ai_assistant = AIVoiceAssistant()
    
    # Initialize Whisper model with error handling and optimized settings
    try:
        whisper_model = initialize_whisper_model(DEFAULT_MODEL_SIZE)
    except Exception as e:
        print(f"Error initializing Whisper model: {str(e)}")
        whisper_model = None
    
    end_time = time.time()
    print(f"Models initialized in {end_time - start_time:.2f} seconds")

# Start initialization in background thread
init_thread = threading.Thread(target=initialize_models)
init_thread.daemon = True
init_thread.start()

# Create recordings directory if it doesn't exist
RECORDINGS_DIR = "static/recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Create images directory if it doesn't exist
IMAGES_DIR = "static/images"
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.join(IMAGES_DIR, "rooms"), exist_ok=True)
os.makedirs(os.path.join(IMAGES_DIR, "facilities"), exist_ok=True)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/rooms')
def rooms():
    return render_template('rooms.html')

@app.route('/facilities')
def facilities():
    return render_template('facilities.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

def safe_remove(file_path, max_attempts=5, delay=0.1):
    """Safely remove a file with multiple attempts"""
    for i in range(max_attempts):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except PermissionError:
            if i < max_attempts - 1:
                time.sleep(delay)
            continue
    return False

def save_conversation(audio_data, assistant_response, transcription):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conversation_dir = os.path.join(RECORDINGS_DIR, timestamp)
    os.makedirs(conversation_dir, exist_ok=True)
    
    # Save guest audio
    guest_audio_path = os.path.join(conversation_dir, "guest_input.wav")
    with open(guest_audio_path, 'wb') as f:
        audio_data.seek(0)
        f.write(audio_data.read())
    
    # Save assistant response audio
    response_audio_path = os.path.join(conversation_dir, "assistant_response.mp3")
    tts = vs.text_to_speech_file(assistant_response, output_path=response_audio_path)
    
    # Save transcription and response text
    transcript_path = os.path.join(conversation_dir, "conversation.txt")
    with open(transcript_path, 'w') as f:
        f.write(f"Guest: {transcription}\n")
        f.write(f"Assistant: {assistant_response}\n")
    
    return conversation_dir

@app.route('/chat', methods=['POST'])
def chat():
    temp_files = []  # Keep track of temporary files
    
    try:
        # Handle voice input only
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_files.append(temp_audio.name)
            audio_file.save(temp_audio.name)
            temp_audio.close()  # Close the file immediately after saving
            
            # Transcribe audio
            if whisper_model is None:
                return jsonify({'error': 'Speech recognition model not available'}), 500
                
            segments, _ = whisper_model.transcribe(temp_audio.name)
            transcription = " ".join([segment.text for segment in segments])
            
            # Get AI response
            response = ai_assistant.interact_with_llm(transcription)
            
            # Save conversation
            conversation_dir = save_conversation(audio_file, response, transcription)
            
            # Convert response to speech
            try:
                audio_response = vs.text_to_speech_file(response)
                temp_files.append(audio_response)
                
                with open(audio_response, 'rb') as f:
                    audio_data = base64.b64encode(f.read()).decode('utf-8')
                
            except Exception as e:
                print(f"Error in text-to-speech: {str(e)}")
                audio_data = None
            
            return jsonify({
                'text': response,
                'audio': audio_data,
                'transcription': transcription,
                'conversation_path': conversation_dir
            })
            
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            safe_remove(temp_file)

@app.route('/recordings')
def list_recordings():
    recordings = []
    for item in os.listdir(RECORDINGS_DIR):
        item_path = os.path.join(RECORDINGS_DIR, item)
        if os.path.isdir(item_path):
            recordings.append({
                'timestamp': item,
                'files': os.listdir(item_path)
            })
    return render_template('recordings.html', recordings=recordings)

def is_silence(data, max_amplitude_threshold=2500):  # Lower threshold to be more sensitive
    """Check if audio data contains silence - optimized version."""
    # Use numpy's faster vectorized operations
    return np.max(np.abs(data)) <= max_amplitude_threshold


def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    """Record audio with optimized processing."""
    frames = []
    chunk_size = 1024
    num_chunks = int(16000 / chunk_size * chunk_length)
    
    # Pre-allocate buffer for better performance
    audio_data = np.zeros(num_chunks * chunk_size, dtype=np.int16)
    
    for i in range(num_chunks):
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)
        # Convert to numpy array in one go at the end instead of per chunk
    
    # Process all frames at once
    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check if the recorded chunk contains silence - optimized with try/finally
    try:
        samplerate, data = wavfile.read(temp_file_path)
        is_silent = is_silence(data)
        return is_silent
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False
    finally:
        # Always try to remove temp file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except:
            pass  # Ignore temp file deletion errors

    

def transcribe_audio(model, file_path):
    """Optimized audio transcription function"""
    # Faster transcription settings
    segments, info = model.transcribe(
        file_path, 
        beam_size=1,               # Greedy decoding (faster)
        language="en",             # Force English for faster processing
        vad_filter=True,           # Voice activity detection to skip silence
        vad_parameters=dict(min_silence_duration_ms=500),  # Optimize VAD
        word_timestamps=False      # Skip word timestamps for speed
    )
    transcription = ' '.join(segment.text for segment in segments)
    return transcription


def initialize_whisper_model(model_size="tiny"):
    """Initialize Whisper model with correct parameters"""
    try:
        # Remove beam_size parameter and use basic configuration
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="./models"  # Cache models locally
        )
        print(f"Successfully initialized Whisper {model_size} model")
        return model
    except Exception as e:
        print(f"Error initializing Whisper model: {str(e)}")
        return None

def download_hotel_images():
    # Dictionary of image URLs from Unsplash (high-quality, free-to-use images)
    image_urls = {
        'hotel-exterior.jpg': 'https://images.unsplash.com/photo-1596386461350-326ccb383e9f?auto=format&fit=crop&w=1600&h=900',
        'hotel-lobby.jpg': 'https://images.unsplash.com/photo-1590381105924-c72589b9ef3f?auto=format&fit=crop&w=1600&h=900',
        'rooms/superior.jpg': 'https://images.unsplash.com/photo-1595576508898-0ad5c879a061?auto=format&fit=crop&w=1600&h=900',
        'rooms/luxury.jpg': 'https://images.unsplash.com/photo-1578683010236-d716f9a3f461?auto=format&fit=crop&w=1600&h=900',
        'rooms/suite.jpg': 'https://images.unsplash.com/photo-1560200353-ce0a76b1d438?auto=format&fit=crop&w=1600&h=900',
        'facilities/restaurant.jpg': 'https://images.unsplash.com/photo-1414235077428-338989a2e8c0?auto=format&fit=crop&w=1600&h=900',
        'facilities/pool.jpg': 'https://images.unsplash.com/photo-1571896349842-33c89424de2d?auto=format&fit=crop&w=1600&h=900',
        'facilities/spa.jpg': 'https://images.unsplash.com/photo-1544161515-4ab6ce6db874?auto=format&fit=crop&w=1600&h=900',
        'facilities/gym.jpg': 'https://images.unsplash.com/photo-1540497077202-7c8a3999166f?auto=format&fit=crop&w=1600&h=900',
        'facilities/heritage.jpg': 'https://images.unsplash.com/photo-1573113617279-5ae16188c10e?auto=format&fit=crop&w=1600&h=900'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for filename, url in image_urls.items():
        filepath = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(filepath):
            try:
                response = requests.get(url, headers=headers, stream=True)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        response.raw.decode_content = True
                        shutil.copyfileobj(response.raw, f)
                    print(f"Downloaded {filename}")
                else:
                    print(f"Failed to download {filename}: Status code {response.status_code}")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

def main():
    print("Taj Mahal Palace Hotel AI Receptionist")
    print("--------------------------------------")
    print("Starting voice recognition system...")
    
    initialization_timeout = 60  # seconds
    start_time = time.time()
    
    # Wait for model initialization with timeout
    while (ai_assistant is None or whisper_model is None) and (time.time() - start_time < initialization_timeout):
        print("Waiting for models to initialize... Press Ctrl+C to start with limited functionality.")
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nStarting with limited functionality...")
            break
    
    # Check if initialization timed out or was interrupted
    if ai_assistant is None or whisper_model is None:
        print("\nWARNING: Models did not initialize completely.")
        print("The system will operate with limited functionality.")
        
        # If Whisper model failed to initialize, create a minimal version
        if whisper_model is None:
            try:
                print("Creating minimal speech recognition model...")
                whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            except Exception as e:
                print(f"Failed to create minimal speech model: {e}")
                print("Speech recognition will not be available.")
                return  # Can't continue without speech recognition
    else:
        print("Models loaded. Ready to interact with guests.")
    
    print("Speak clearly to interact with the AI receptionist")
    print("Press Ctrl+C to exit")
    print("--------------------------------------")
    
    # Prepare audio settings for faster performance
    try:
        audio = pyaudio.PyAudio()
        
        # Optimize stream settings for better performance
        stream = audio.open(
            format=pyaudio.paInt16, 
            channels=1, 
            rate=16000, 
            input=True, 
            frames_per_buffer=1024,
            input_device_index=None,  # Use default input device
            start=True  # Start the stream immediately
        )
        
        # Pre-allocate buffer for guest input
        guest_input_transcription = ""
        
        # Create a memory buffer to avoid excessive file operations
        memory_buffer = {"is_processing": False}
        
        # Optimize query handling with a simple thread pool
        executor = ThreadPoolExecutor(max_workers=2)
        
        def process_audio_response(transcription):
            """Process transcription and generate response in a separate thread"""
            nonlocal guest_input_transcription
            nonlocal memory_buffer
            
            print("Guest: {}".format(transcription))
            
            # Add guest input to transcript efficiently
            guest_input_transcription += "Guest: " + transcription + "\n"
            
            try:
                # Measure response time
                start_time = time.time()
                
                # Process guest input and get response from AI assistant
                if ai_assistant is not None:
                    output = ai_assistant.interact_with_llm(transcription)
                else:
                    # Fallback if assistant is not available
                    output = "Taj Mahal Palace Mumbai Assistant: I apologize, but our AI system is currently initializing. How may I assist you with basic information?"
                
                end_time = time.time()
                print(f"Response generated in {end_time - start_time:.2f} seconds")
                
                if output:
                    output = output.lstrip()
                    try:
                        vs.play_text_to_speech(output)
                    except Exception as e:
                        print(f"Error in text-to-speech: {e}")
                    print("Taj Mahal Palace Hotel Receptionist: {}".format(output))
            except Exception as e:
                print(f"Error processing response: {e}")
                try:
                    fallback = "Taj Mahal Palace Mumbai Assistant: I apologize for the technical difficulty. How else may I assist you?"
                    vs.play_text_to_speech(fallback)
                    print(fallback)
                except:
                    pass
            finally:
                # Mark processing as complete
                memory_buffer["is_processing"] = False

        try:
            while True:
                # Skip if we're currently processing audio
                if memory_buffer["is_processing"]:
                    time.sleep(0.1)
                    continue
                    
                chunk_file = "temp_audio_chunk.wav"
                
                # Record audio chunk
                print("_")
                try:
                    is_silent = record_audio_chunk(audio, stream)
                    
                    if not is_silent and os.path.exists(chunk_file) and whisper_model is not None:
                        # Mark as processing to avoid parallel transcription
                        memory_buffer["is_processing"] = True
                        
                        # Transcribe audio
                        try:
                            transcription = transcribe_audio(whisper_model, chunk_file)
                            
                            # Process in background thread
                            executor.submit(process_audio_response, transcription)
                        except Exception as e:
                            print(f"Error transcribing audio: {e}")
                            memory_buffer["is_processing"] = False
                except Exception as e:
                    print(f"Error in audio processing: {e}")
                    time.sleep(0.5)  # Short delay to prevent tight error loop
        
        except KeyboardInterrupt:
            print("\nStopping the Taj Mahal Palace Hotel AI Receptionist...")

        finally:
            # Clean up resources
            try:
                stream.stop_stream()
                stream.close()
                audio.terminate()
                executor.shutdown(wait=False)
            except Exception as e:
                print(f"Error during cleanup: {e}")
    
    except Exception as e:
        print(f"Error initializing audio: {e}")
        print("Unable to start audio capture. Please check your microphone settings.")

if __name__ == "__main__":
    download_hotel_images()
    app.run(debug=True)
