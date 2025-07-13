"""
Voice processing module for voice input/output capabilities
"""
import os
import asyncio
import tempfile
import wave
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

import speech_recognition as sr
import pyttsx3
import pyaudio
from loguru import logger

from models import VoiceInput, VoiceCommand, VoiceRequest
from config.settings import settings


class VoiceProcessor:
    """Voice processing for input/output operations"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = None
        self.audio = pyaudio.PyAudio()
        self.initialize_tts()
        
        # Voice command patterns
        self.command_patterns = {
            "start_scraping": ["scrape", "start scraping", "begin scraping", "fetch content"],
            "process_content": ["process", "process content", "ai process", "transform"],
            "start_iteration": ["iterate", "start iteration", "human review", "feedback"],
            "approve_content": ["approve", "approve content", "accept", "finalize"],
            "reject_content": ["reject", "reject content", "decline", "return"],
            "search_content": ["search", "find", "look for", "query"]
        }
        
        # Performance tracking
        self.voice_stats = {
            "total_voice_inputs": 0,
            "successful_transcriptions": 0,
            "total_voice_outputs": 0,
            "average_confidence": 0.0
        }
    
    def initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a good voice
                for voice in voices:
                    if "en" in voice.languages[0].lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', settings.voice_rate)
            self.tts_engine.setProperty('volume', 0.9)
            
            logger.info("Text-to-speech engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.tts_engine = None
    
    async def process_voice_input(self, request: VoiceRequest) -> VoiceInput:
        """Process voice input and transcribe to text"""
        start_time = datetime.utcnow()
        
        try:
            # Transcribe audio file
            transcribed_text, confidence = await self._transcribe_audio(request.audio_file_path)
            
            # Detect command type
            detected_command = self._detect_command(transcribed_text)
            
            # Create voice input object
            voice_input = VoiceInput(
                command=detected_command or VoiceCommand.SEARCH_CONTENT,
                audio_file_path=request.audio_file_path,
                transcribed_text=transcribed_text,
                confidence_score=confidence,
                processed_at=start_time
            )
            
            # Update statistics
            self._update_voice_stats(True, confidence)
            
            logger.info(f"Voice input processed: {detected_command} (confidence: {confidence:.3f})")
            return voice_input
            
        except Exception as e:
            logger.error(f"Voice input processing failed: {e}")
            self._update_voice_stats(False, 0.0)
            raise
    
    async def _transcribe_audio(self, audio_file_path: str) -> tuple[str, float]:
        """Transcribe audio file to text"""
        try:
            with sr.AudioFile(audio_file_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record audio
                audio = self.recognizer.record(source)
                
                # Transcribe using Google Speech Recognition
                text = self.recognizer.recognize_google(
                    audio,
                    language=settings.voice_language
                )
                
                # Get confidence score (Google doesn't provide this directly)
                # We'll use a heuristic based on audio quality
                confidence = self._calculate_confidence_score(audio_file_path)
                
                return text, confidence
                
        except sr.UnknownValueError:
            raise ValueError("Speech could not be understood")
        except sr.RequestError as e:
            raise ValueError(f"Speech recognition service error: {e}")
        except Exception as e:
            raise ValueError(f"Audio transcription failed: {e}")
    
    def _calculate_confidence_score(self, audio_file_path: str) -> float:
        """Calculate confidence score based on audio quality"""
        try:
            with wave.open(audio_file_path, 'rb') as wav_file:
                # Get audio parameters
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                
                # Read audio data
                audio_data = wav_file.readframes(frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Calculate signal-to-noise ratio (simplified)
                signal_power = np.mean(audio_array ** 2)
                noise_power = np.var(audio_array)
                
                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                    # Normalize SNR to 0-1 range
                    confidence = min(max((snr + 20) / 40, 0.0), 1.0)
                else:
                    confidence = 0.5
                
                # Adjust based on duration (longer is better, up to a point)
                duration_factor = min(duration / 5.0, 1.0)
                confidence = (confidence * 0.7 + duration_factor * 0.3)
                
                return confidence
                
        except Exception as e:
            logger.warning(f"Could not calculate confidence score: {e}")
            return 0.5
    
    def _detect_command(self, text: str) -> Optional[VoiceCommand]:
        """Detect voice command from transcribed text"""
        text_lower = text.lower()
        
        for command, patterns in self.command_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return VoiceCommand(command)
        
        return None
    
    def _update_voice_stats(self, success: bool, confidence: float):
        """Update voice processing statistics"""
        self.voice_stats["total_voice_inputs"] += 1
        if success:
            self.voice_stats["successful_transcriptions"] += 1
        
        # Update average confidence
        total_successful = self.voice_stats["successful_transcriptions"]
        current_avg = self.voice_stats["average_confidence"]
        self.voice_stats["average_confidence"] = (
            (current_avg * (total_successful - 1) + confidence) / total_successful
        )
    
    async def speak_text(self, text: str, save_to_file: bool = False) -> Optional[str]:
        """Convert text to speech"""
        if not self.tts_engine:
            logger.warning("TTS engine not available")
            return None
        
        try:
            if save_to_file:
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=False,
                    dir="./temp"
                )
                temp_file.close()
                
                # Configure TTS to save to file
                self.tts_engine.save_to_file(text, temp_file.name)
                self.tts_engine.runAndWait()
                
                self.voice_stats["total_voice_outputs"] += 1
                logger.info(f"Speech saved to file: {temp_file.name}")
                return temp_file.name
            else:
                # Speak directly
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                
                self.voice_stats["total_voice_outputs"] += 1
                logger.info("Speech output completed")
                return None
                
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            return None
    
    async def record_audio(self, duration: int = 5, sample_rate: int = 16000) -> str:
        """Record audio from microphone"""
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                dir="./temp"
            )
            temp_file.close()
            
            # Configure audio stream
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            logger.info(f"Recording audio for {duration} seconds...")
            
            frames = []
            for _ in range(0, int(sample_rate / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            # Save to WAV file
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(b''.join(frames))
            
            logger.info(f"Audio recording saved: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Audio recording failed: {e}")
            raise
    
    def get_voice_stats(self) -> Dict[str, Any]:
        """Get voice processing statistics"""
        return self.voice_stats.copy()
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.audio:
            self.audio.terminate()
        if self.tts_engine:
            self.tts_engine.stop()


class VoiceCommandProcessor:
    """Process voice commands and execute corresponding actions"""
    
    def __init__(self, workflow_manager=None):
        self.voice_processor = VoiceProcessor()
        self.workflow_manager = workflow_manager
        
        # Command handlers
        self.command_handlers = {
            VoiceCommand.START_SCRAPING: self._handle_start_scraping,
            VoiceCommand.PROCESS_CONTENT: self._handle_process_content,
            VoiceCommand.START_ITERATION: self._handle_start_iteration,
            VoiceCommand.APPROVE_CONTENT: self._handle_approve_content,
            VoiceCommand.REJECT_CONTENT: self._handle_reject_content,
            VoiceCommand.SEARCH_CONTENT: self._handle_search_content
        }
    
    async def process_voice_command(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Process voice command and return response"""
        try:
            # Get command handler
            handler = self.command_handlers.get(voice_input.command)
            if not handler:
                return {
                    "success": False,
                    "message": f"Unknown command: {voice_input.command}",
                    "data": None
                }
            
            # Execute command
            result = await handler(voice_input)
            
            # Generate voice response
            response_text = self._generate_response_text(result)
            await self.voice_processor.speak_text(response_text)
            
            return result
            
        except Exception as e:
            logger.error(f"Voice command processing failed: {e}")
            error_response = f"Sorry, I couldn't process that command. Error: {str(e)}"
            await self.voice_processor.speak_text(error_response)
            
            return {
                "success": False,
                "message": error_response,
                "data": None
            }
    
    async def _handle_start_scraping(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle start scraping command"""
        # Extract URL from voice input (simplified)
        # In a real implementation, you'd use NLP to extract the URL
        response_text = "Starting content scraping. Please provide the URL to scrape."
        await self.voice_processor.speak_text(response_text)
        
        return {
            "success": True,
            "message": "Scraping initiated",
            "data": {"action": "start_scraping", "requires_url": True}
        }
    
    async def _handle_process_content(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle process content command"""
        response_text = "Starting AI content processing. This may take a few moments."
        await self.voice_processor.speak_text(response_text)
        
        return {
            "success": True,
            "message": "Content processing initiated",
            "data": {"action": "process_content"}
        }
    
    async def _handle_start_iteration(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle start iteration command"""
        response_text = "Starting human-in-the-loop iteration. Please review the content."
        await self.voice_processor.speak_text(response_text)
        
        return {
            "success": True,
            "message": "Iteration started",
            "data": {"action": "start_iteration"}
        }
    
    async def _handle_approve_content(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle approve content command"""
        response_text = "Content approved. Moving to finalization stage."
        await self.voice_processor.speak_text(response_text)
        
        return {
            "success": True,
            "message": "Content approved",
            "data": {"action": "approve_content"}
        }
    
    async def _handle_reject_content(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle reject content command"""
        response_text = "Content rejected. Returning for revision."
        await self.voice_processor.speak_text(response_text)
        
        return {
            "success": True,
            "message": "Content rejected",
            "data": {"action": "reject_content"}
        }
    
    async def _handle_search_content(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle search content command"""
        # Extract search query from voice input
        search_query = voice_input.transcribed_text.replace("search", "").replace("find", "").strip()
        
        response_text = f"Searching for: {search_query}"
        await self.voice_processor.speak_text(response_text)
        
        return {
            "success": True,
            "message": "Search initiated",
            "data": {"action": "search_content", "query": search_query}
        }
    
    def _generate_response_text(self, result: Dict[str, Any]) -> str:
        """Generate response text for voice output"""
        if result["success"]:
            return f"Command completed successfully. {result['message']}"
        else:
            return f"Command failed. {result['message']}"
    
    def get_voice_stats(self) -> Dict[str, Any]:
        """Get voice processing statistics"""
        return self.voice_processor.get_voice_stats()
    
    def cleanup(self):
        """Clean up voice processor"""
        self.voice_processor.cleanup()


# Global voice processor instance
voice_processor = VoiceProcessor() 