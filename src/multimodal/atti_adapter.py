"""
ATTI Multimodal Adapter
Integrates avatar framework with ATTI ecosystem
"""

import logging
from typing import Dict, Optional
import requests

logger = logging.getLogger(__name__)


class ATTIMultimodalAdapter:
    """Adapter for ATTI integration"""
    
    def __init__(self, atti_api_url: str = "http://localhost:3001"):
        """
        Initialize ATTI adapter
        
        Args:
            atti_api_url: ATTI API base URL
        """
        self.atti_api_url = atti_api_url
        
    def get_persona(self, persona_id: str) -> Dict:
        """
        Get persona from ATTI PersonaManager
        
        Args:
            persona_id: Persona identifier
            
        Returns:
            Persona configuration
        """
        try:
            response = requests.get(
                f"{self.atti_api_url}/api/personas/{persona_id}",
                timeout=5,
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get persona: {e}")
            return {}
            
    def get_voice_config(self, voice_id: str) -> Dict:
        """
        Get voice configuration from ATTI Voice Layer
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Voice configuration
        """
        try:
            response = requests.get(
                f"{self.atti_api_url}/api/voices/{voice_id}",
                timeout=5,
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get voice config: {e}")
            return {}
            
    def publish_video(self, video_path: str, metadata: Dict) -> Dict:
        """
        Publish generated video to ATTI social publishers
        
        Args:
            video_path: Path to video file
            metadata: Video metadata
            
        Returns:
            Publication result
        """
        try:
            with open(video_path, "rb") as f:
                files = {"video": f}
                response = requests.post(
                    f"{self.atti_api_url}/api/social/publish",
                    files=files,
                    data=metadata,
                    timeout=30,
                )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to publish video: {e}")
            return {"status": "error", "message": str(e)}
            
    def log_metrics(self, metrics: Dict) -> None:
        """
        Log metrics to ATTI monitoring
        
        Args:
            metrics: Metrics dictionary
        """
        try:
            requests.post(
                f"{self.atti_api_url}/api/metrics",
                json=metrics,
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
