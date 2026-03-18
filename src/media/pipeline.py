"""
Media Pipeline Orchestrator
Coordinates TTS, Avatar, and Video generation
"""

import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MediaPipeline:
    """Orchestrates the complete media generation pipeline"""
    
    def __init__(self, tts_engine, avatar_engine):
        """
        Initialize media pipeline
        
        Args:
            tts_engine: TTS engine instance
            avatar_engine: Avatar engine instance
        """
        self.tts = tts_engine
        self.avatar = avatar_engine
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_video(
        self,
        text: str,
        avatar: str = "sofia",
        voice: str = "sofia",
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Generate complete video with avatar and voice
        
        Args:
            text: Text to synthesize
            avatar: Avatar to use
            voice: Voice to use
            output_path: Custom output path
            
        Returns:
            Dictionary with video metadata
        """
        logger.info(f"Starting video generation: {text[:50]}...")
        
        # Step 1: Synthesize speech
        audio, sample_rate = self.tts.synthesize(text, voice=voice)
        duration = len(audio) / sample_rate
        
        logger.info(f"TTS complete: {duration:.2f}s audio")
        
        # Step 2: Generate avatar animation
        animation = self.avatar.generate_animation(
            avatar_name=avatar,
            audio_path="temp_audio.wav",
            duration=duration,
        )
        
        logger.info(f"Avatar animation complete: {animation['frames']} frames")
        
        # Step 3: Combine audio and video
        output_file = output_path or self.output_dir / f"{avatar}_video.mp4"
        
        result = {
            "text": text,
            "avatar": avatar,
            "voice": voice,
            "duration": duration,
            "video_path": str(output_file),
            "audio_sample_rate": sample_rate,
            "status": "complete",
        }
        
        logger.info(f"Video generation complete: {output_file}")
        
        return result
        
    def generate_batch(self, items: list) -> list:
        """
        Generate multiple videos in batch
        
        Args:
            items: List of {text, avatar, voice} dicts
            
        Returns:
            List of generated video metadata
        """
        results = []
        for item in items:
            result = self.generate_video(**item)
            results.append(result)
        return results
