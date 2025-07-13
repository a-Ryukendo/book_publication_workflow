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
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = None
        self.audio = pyaudio.PyAudio()
        self._init_tts()
        self.command_patterns = {
            "start_scraping": ["scrape", "start scraping", "begin scraping", "fetch content"],
            "process_content": ["process", "process content", "ai process", "transform"],
            "start_iteration": ["iterate", "start iteration", "human review", "feedback"],
            "approve_content": ["approve", "approve content", "accept", "finalize"],
            "reject_content": ["reject", "reject content", "decline", "return"],
            "search_content": ["search", "find", "look for", "query"]
        }
        self.voice_stats = {
            "total_voice_inputs": 0,
            "successful_transcriptions": 0,
            "total_voice_outputs": 0,
            "average_confidence": 0.0
        }

    def _init_tts(self):
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if "en" in voice.languages[0].lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            self.tts_engine.setProperty('rate', settings.voice_rate)
            self.tts_engine.setProperty('volume', 0.9)
            logger.info("TTS engine ready")
        except Exception as e:
            logger.error(f"TTS init failed: {e}")
            self.tts_engine = None

    async def process_voice_input(self, request: VoiceRequest) -> VoiceInput:
        start = datetime.utcnow()
        try:
            text, conf = await self._transcribe_audio(request.audio_file_path)
            cmd = self._detect_command(text)
            voice_input = VoiceInput(
                command=cmd or VoiceCommand.SEARCH_CONTENT,
                audio_file_path=request.audio_file_path,
                transcribed_text=text,
                confidence_score=conf,
                processed_at=start
            )
            self._update_voice_stats(True, conf)
            logger.info(f"Voice input: {cmd} (conf: {conf:.3f})")
            return voice_input
        except Exception as e:
            logger.error(f"Voice input failed: {e}")
            self._update_voice_stats(False, 0.0)
            raise

    async def _transcribe_audio(self, audio_file_path: str) -> tuple[str, float]:
        try:
            with sr.AudioFile(audio_file_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language=settings.voice_language)
                conf = self._calc_conf(audio_file_path)
                return text, conf
        except sr.UnknownValueError:
            raise ValueError("Speech not understood")
        except sr.RequestError as e:
            raise ValueError(f"Speech service error: {e}")
        except Exception as e:
            raise ValueError(f"Transcription failed: {e}")

    def _calc_conf(self, audio_file_path: str) -> float:
        try:
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                audio_data = wav_file.readframes(frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                signal_power = np.mean(audio_array ** 2)
                noise_power = np.var(audio_array)
                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                    conf = min(max((snr + 20) / 40, 0.0), 1.0)
                else:
                    conf = 0.5
                duration_factor = min(duration / 5.0, 1.0)
                conf = (conf * 0.7 + duration_factor * 0.3)
                return conf
        except Exception as e:
            logger.warning(f"Conf calc failed: {e}")
            return 0.5

    def _detect_command(self, text: str) -> Optional[VoiceCommand]:
        text_lower = text.lower()
        for command, patterns in self.command_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return VoiceCommand(command)
        return None

    def _update_voice_stats(self, success: bool, conf: float):
        self.voice_stats["total_voice_inputs"] += 1
        if success:
            self.voice_stats["successful_transcriptions"] += 1
        total = self.voice_stats["successful_transcriptions"]
        avg = self.voice_stats["average_confidence"]
        self.voice_stats["average_confidence"] = ((avg * (total - 1) + conf) / total)

    async def speak_text(self, text: str, save_to_file: bool = False) -> Optional[str]:
        if not self.tts_engine:
            logger.warning("TTS not available")
            return None
        try:
            if save_to_file:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="./temp")
                temp_file.close()
                self.tts_engine.save_to_file(text, temp_file.name)
                self.tts_engine.runAndWait()
                self.voice_stats["total_voice_outputs"] += 1
                logger.info(f"Speech saved: {temp_file.name}")
                return temp_file.name
            else:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                self.voice_stats["total_voice_outputs"] += 1
                logger.info("Speech output done")
                return None
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return None

    async def record_audio(self, duration: int = 5, sample_rate: int = 16000) -> str:
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="./temp")
            temp_file.close()
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            logger.info(f"Recording {duration}s...")
            frames = []
            for _ in range(0, int(sample_rate / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)
            stream.stop_stream()
            stream.close()
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(sample_rate)
                wf.writeframes(b''.join(frames))
            self.voice_stats["total_voice_inputs"] += 1
            logger.info(f"Audio recorded: {temp_file.name}")
            return temp_file.name
        except Exception as e:
            logger.error(f"Record failed: {e}")
            return ""

    def get_voice_stats(self) -> Dict[str, Any]:
        return self.voice_stats.copy()

    def cleanup(self):
        try:
            self.audio.terminate()
            if self.tts_engine:
                self.tts_engine.stop()
            logger.info("Voice resources cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

class VoiceCommandProcessor:
    def __init__(self, workflow_manager=None):
        self.workflow = workflow_manager

    async def process_voice_command(self, voice_input: VoiceInput) -> Dict[str, Any]:
        cmd = voice_input.command.value if isinstance(voice_input.command, VoiceCommand) else str(voice_input.command)
        if cmd == "start_scraping":
            return await self._handle_start_scraping(voice_input)
        if cmd == "process_content":
            return await self._handle_process_content(voice_input)
        if cmd == "start_iteration":
            return await self._handle_start_iteration(voice_input)
        if cmd == "approve_content":
            return await self._handle_approve_content(voice_input)
        if cmd == "reject_content":
            return await self._handle_reject_content(voice_input)
        if cmd == "search_content":
            return await self._handle_search_content(voice_input)
        return {"success": False, "message": "Unknown command", "data": {}}

    async def _handle_start_scraping(self, voice_input: VoiceInput) -> Dict[str, Any]:
        # Example: implement actual logic as needed
        return {"success": True, "message": "Started scraping", "data": {}}

    async def _handle_process_content(self, voice_input: VoiceInput) -> Dict[str, Any]:
        return {"success": True, "message": "Processing content", "data": {}}

    async def _handle_start_iteration(self, voice_input: VoiceInput) -> Dict[str, Any]:
        return {"success": True, "message": "Started iteration", "data": {}}

    async def _handle_approve_content(self, voice_input: VoiceInput) -> Dict[str, Any]:
        return {"success": True, "message": "Content approved", "data": {}}

    async def _handle_reject_content(self, voice_input: VoiceInput) -> Dict[str, Any]:
        return {"success": True, "message": "Content rejected", "data": {}}

    async def _handle_search_content(self, voice_input: VoiceInput) -> Dict[str, Any]:
        return {"success": True, "message": "Search done", "data": {}}

    def get_voice_stats(self) -> Dict[str, Any]:
        return {}

    def cleanup(self):
        pass 