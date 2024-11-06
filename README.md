AI Podcast Generator

This project uses AI to generate podcasts based on the content of a provided PDF document. It combines Gradio for the interface, Hugging Face's Transformers library for speech recognition, and gTTS for text-to-speech. With a single PDF input, the system creates a conversational podcast script and generates an audio podcast in a few simple steps.
Features

    PDF-to-Text Extraction: Extracts text content from a PDF file.
    Text Generation: Uses a local AI model (e.g., Ollama) to generate a podcast script from extracted text.
    Text-to-Speech Conversion: Converts generated script to speech.
    Gradio Interface: Interactive web interface for user input and podcast generation.
    Configurable Options: Choose podcast duration, number of participants, and AI model to customize the generated podcast.

Requirements

    Python 3.7+
    Libraries: gradio, torch, transformers, pydub, PyPDF2, gtts, requests, numpy, and ffmpeg (for audio processing).

Installation and Setup

To set up and run the AI Podcast Generator, use the provided install_and_run.sh script for one-click installation.
Steps

    Place install.sh in the same directory as podcastaiworking.py.
    Run the script with:

    bash

    chmod +x install.sh
    ./install.sh

The script will:

    Update system dependencies
    Install required Python packages
    Start the Gradio application and serve it on http://0.0.0.0:7860

Usage

    Open the Gradio interface in your browser at http://0.0.0.0:7860.
    Upload a PDF document for conversion.
    Configure podcast settings:
        Podcast Duration: Choose the desired podcast length in minutes.
        Participants: Select the number of participants in the conversation.
        Model: Choose a local AI model (e.g., Ollama or Mistral).
    Click Generate Podcast to start the conversion process.

Once completed, the interface will display:

    A downloadable Transcript of the podcast.
    The Generated Podcast audio for playback.

Project Structure

    podcastaiworking.py: Main script for AI Podcast Generator.
    install.sh: Installation and setup script for one-click deployment.
    requirements.txt: List of Python dependencies.

Troubleshooting

If you encounter a ModuleNotFoundError, make sure all required Python packages are installed. Re-run install_and_run.sh to ensure dependencies are correctly set up.
License

This project is licensed under the MIT License.
