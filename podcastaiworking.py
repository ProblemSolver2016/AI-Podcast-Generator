import gradio as gr
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
import numpy as np
import PyPDF2
import os
import tempfile
import logging
from typing import Optional, Tuple
import requests
import json
from pathlib import Path
from gtts import gTTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PodcastGenerator:
    def __init__(self):
        try:
            # Initialize speech recognition model
            self.model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-base-960h",
                ignore_mismatched_sizes=True
            )
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            
            # Get available Ollama models
            self.available_models = self.get_available_models()
            logger.info(f"Available models: {self.available_models}")
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def get_available_models(self) -> list:
        """
        Get list of available models from Ollama
        """
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json()['models']]
                return models if models else ["mistral"]  # Default to mistral if no models found
            return ["mistral"]
        except:
            logger.warning("Could not fetch Ollama models, defaulting to mistral")
            return ["mistral"]

    def generate_text_from_local_ai(self, prompt: str, model_name: str) -> str:
        """
        Generate text using local Ollama model
        """
        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            try:
                response = requests.post(url, json=data, timeout=60)
                response.raise_for_status()
                return response.json()['response']
            except requests.exceptions.RequestException as e:
                raise Exception(f"Ollama API error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error generating text with {model_name}: {str(e)}")
            raise

    def text_to_speech(self, text: str, output_path: str) -> str:
        """
        Convert text to speech using gTTS
        """
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            return output_path
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_file: str) -> str:
        """
        Extract text from PDF file
        """
        try:
            text = ""
            with open(pdf_file, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise

    def generate_podcast_script(self, pdf_text: str, time_limit: int, num_participants: int) -> str:
        """
        Generate a podcast script based on the PDF content
        """
        target_word_count = time_limit * 150
        
        prompt = f"""
        Create an engaging {time_limit}-minute podcast script discussing the following article. 
        Include {num_participants} distinct speakers with clear personalities.
        Target approximately {target_word_count} words total.
        Make it conversational, informative, and engaging while covering the key points.
        
        Format the output as:
        [Speaker Name]: Dialog text
        
        Use speaker names like Host, Expert1, Expert2, etc.
        
        Article content:
        {pdf_text[:3000]}
        """
        return prompt.strip()

    def generate_podcast(
        self,
        pdf_file: str,
        time_limit: int,
        num_participants: int,
        model_name: str,
        progress=gr.Progress()
    ) -> Tuple[str, str, tuple]:
        """
        Generate podcast from PDF
        """
        try:
            if not pdf_file:
                raise ValueError("Please upload a PDF file")

            # Extract text from PDF
            progress(0.1, desc="Reading PDF...")
            pdf_text = self.extract_text_from_pdf(pdf_file)
            if not pdf_text.strip():
                raise ValueError("No text content found in PDF")

            # Generate podcast script using local AI
            progress(0.3, desc="Generating script...")
            prompt = self.generate_podcast_script(pdf_text, time_limit, num_participants)
            script = self.generate_text_from_local_ai(prompt, model_name)

            # Create temporary directory for audio files
            temp_dir = tempfile.mkdtemp()
            
            # Convert script to audio
            progress(0.6, desc="Converting to speech...")
            audio_path = os.path.join(temp_dir, "podcast.mp3")
            self.text_to_speech(script, audio_path)

            # Save transcript
            progress(0.8, desc="Saving transcript...")
            transcript_path = os.path.join(temp_dir, "transcript.txt")
            with open(transcript_path, "w", encoding='utf-8') as f:
                f.write(script)

            progress(1.0, desc="Done!")
            # Return paths and load audio for playback
            audio_segment = AudioSegment.from_file(audio_path)
            return (
                "Podcast generated successfully!",
                transcript_path,
                (audio_segment.frame_rate, np.array(audio_segment.get_array_of_samples()))
            )

        except Exception as e:
            error_msg = f"Error generating podcast: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", (16000, np.zeros(16000))

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    generator = PodcastGenerator()
    
    with gr.Blocks() as interface:
        gr.Markdown("# AI Podcast Generator")
        gr.Markdown("Upload a PDF to generate an AI-powered podcast discussion about its content.")
        
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(
                    label="Upload PDF (required)",
                    file_types=[".pdf"]
                )
                
                time_limit = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Podcast Duration (minutes)"
                )
                
                num_participants = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="Number of Participants"
                )
                
                model_choice = gr.Dropdown(
                    choices=generator.available_models,
                    value=generator.available_models[0],
                    label="Select Local AI Model"
                )
                
                generate_btn = gr.Button("Generate Podcast", variant="primary")
            
            with gr.Column():
                status_output = gr.Textbox(label="Status")
                transcript_output = gr.File(label="Transcript")
                audio_output = gr.Audio(label="Generated Podcast")
        
        generate_btn.click(
            fn=generator.generate_podcast,
            inputs=[
                pdf_input,
                time_limit,
                num_participants,
                model_choice
            ],
            outputs=[status_output, transcript_output, audio_output]
        )
    
    return interface

if __name__ == "__main__":
    logger.info("Starting AI Podcast Generator")
    logger.info(f"Gradio version: {gr.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    iface = create_gradio_interface()
    iface.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860
    )
