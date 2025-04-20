import os
import time
import pygame
from gtts import gTTS
import tempfile
import threading
from functools import lru_cache

# Initialize pygame mixer once at module level for better performance
pygame.mixer.init()

@lru_cache(maxsize=20)  # Cache recent TTS for repeated phrases
def text_to_speech_cached(text, language='en', slow=False):
    """Cached version of text to speech conversion"""
    tts = gTTS(text=text, lang=language, slow=slow)
    
    # Use a more efficient temp file approach
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_audio_file = temp_file.name
    
    # Save the audio
    tts.save(temp_audio_file)
    return temp_audio_file

def play_text_to_speech(text, language='en', slow=False):
    """Optimized text to speech playback"""
    # For very short responses, use pre-cached versions when possible
    if len(text) < 100:
        temp_audio_file = text_to_speech_cached(text, language, slow)
    else:
        # Generate fresh TTS for longer, unique responses
        tts = gTTS(text=text, lang=language, slow=slow)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_audio_file = temp_file.name
        tts.save(temp_audio_file)
    
    # Play the audio more efficiently
    try:
        # Load and play in a non-blocking manner
        pygame.mixer.music.load(temp_audio_file)
        pygame.mixer.music.play()
        
        # Wait for playback to complete using a more efficient approach
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(30)  # Higher tick rate for more responsive exit
        
        pygame.mixer.music.stop()
    except Exception as e:
        print(f"Error during audio playback: {e}")
    finally:
        # Clean up in a separate thread to avoid blocking
        def delayed_cleanup():
            time.sleep(0.5)  # Small delay to ensure file is not in use
            try:
                os.remove(temp_audio_file)
            except:
                pass  # Ignore cleanup errors
                
        cleanup_thread = threading.Thread(target=delayed_cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()

def text_to_speech_file(text, output_path=None, language='en', slow=False):
    """Generate TTS and save to a file, optimized for performance"""
    tts = gTTS(text=text, lang=language, slow=slow)
    
    if not output_path:
        # Create a temp file if no output path specified
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            output_path = temp_file.name
    
    tts.save(output_path)
    return output_path
