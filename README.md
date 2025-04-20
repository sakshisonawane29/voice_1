# AI Voice Assistant Receptionist for Metro Luxe Hotel

This repository contains code for a voice assistant that serves as a professional receptionist for Metro Luxe Hotel. The assistant is designed to record audio input from guests, transcribe it, and then interact with the AI model to provide relevant responses about the hotel services and offerings.

## Features

- Record audio input from hotel guests in chunks
- Transcribe the recorded audio using a pre-trained Whisper model
- Interact with Mistral LLM via Hugging Face API to generate responses
- Utilizes a RAG (Retrieval-Augmented Generation) pipeline with hotel knowledge base
- Maintains conversation memory to provide contextual responses

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.8+
- `pyaudio`
- `numpy`
- `faster_whisper` (for speech recognition)
- `qdrant_client` (for vector database)
- Hugging Face API key (for access to Mistral LLM)
- Other dependencies specified in `requirements.txt`

## Setup

1. Clone this repository to your local machine.

   ```bash
   git clone https://github.com/yourusername/hotel_voice_assistant.git
   ```

2. Install the dependencies using pip.

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face API key:
   - Sign up on Hugging Face and obtain an API key
   - Add your API key in the `AIVoiceAssistant.py` file or set it as an environment variable

4. Start a local Qdrant instance (or use a cloud-hosted one) and update the URL if needed.

## Usage

1. Run the main script app.py.

   ```bash
   python app.py
   ```

2. The voice assistant will start listening for guest inquiries
3. Speak into the microphone when prompted
4. The assistant will transcribe your speech, process it through the RAG pipeline, and respond with relevant hotel information

## System Architecture

- **Speech Recognition**: Uses faster_whisper for transcribing guest speech
- **Language Model**: Mistral LLM accessed via Hugging Face Inference API
- **Vector Database**: Qdrant for storing and retrieving vector embeddings
- **Knowledge Base**: Hotel information stored in vector format for contextual retrieval
- **Memory**: Conversation history stored with a 1500 token limit

## Configuration

- You can adjust the default model size and chunk length in the script as per your requirements
- Modify the system prompt in the `AIVoiceAssistant` class to change the assistant's behavior
- Update the hotel knowledge base (`hotel.txt`) with current information as needed

