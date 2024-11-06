#!/bin/bash

# Script to install dependencies and start the AI Podcast Generator

echo "Starting installation of AI Podcast Generator..."

# Update and install system dependencies
echo "Updating system and installing dependencies..."
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg

# Navigate to the project directory
# Ensure you're in the directory where the script resides
PROJECT_DIR="$(dirname "$0")"
cd "$PROJECT_DIR" || exit

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install gradio torch transformers pydub PyPDF2 gtts requests numpy

# Run local AI server (Ollama)
echo "Starting the local AI server (Ollama)..."
# Placeholder: Adjust the command to start Ollama as needed
# Example for Docker: docker run -d -p 11434:11434 ollama/ollama

# Start the Gradio app
echo "Launching the AI Podcast Generator..."
python podcastaiworking.py  # Make sure this is the correct script filename

