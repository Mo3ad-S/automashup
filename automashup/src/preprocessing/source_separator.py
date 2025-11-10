"""
Enhanced source separation module with quality assessment and fallback options.
"""
import numpy as np
import allin1
import os
from typing import Dict, Any, Optional, List
from automashup.src.preprocessing.quality_assessment import QualityAssessment


class SourceSeparator:
    """Enhanced source separator with quality assessment."""
    
    def __init__(self, stored_data_path: str = ".", use_gpu: bool = True):
        self.stored_data_path = stored_data_path
        self.use_gpu = use_gpu
        self.quality_assessor = QualityAssessment()
    
    def separate(self, song_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Separate audio into stems with quality assessment.
        
        Args:
            song_path: Path to input audio file
            overwrite: Whether to overwrite existing separations
        
        Returns:
            Dictionary with separation results and quality metrics
        """
        # Use allin1 for separation (Demucs)
        struct_dir = f'{self.stored_data_path}/struct'
        demix_dir = f'{self.stored_data_path}/separated'
        
        # Analyze and separate
        allin1.analyze(
            song_path, 
            out_dir=struct_dir, 
            demix_dir=demix_dir, 
            keep_byproducts=True, 
            overwrite=overwrite
        )
        
        # Extract filename
        filename = os.path.splitext(os.path.basename(song_path))[0]
        
        # Assess quality of each separated stem
        stems = ['vocals', 'bass', 'drums', 'other']
        quality_results = {}
        
        for stem in stems:
            stem_path = f"{demix_dir}/htdemucs/{filename}/{stem}.wav"
            if os.path.exists(stem_path):
                # Load separated audio
                import librosa
                stem_audio, sr = librosa.load(stem_path, sr=None)
                
                # Assess quality
                quality = self.quality_assessor.assess_separation_quality(stem_audio)
                quality_results[stem] = {
                    'path': stem_path,
                    'quality': quality,
                    'sr': sr
                }
        
        return {
            'filename': filename,
            'stems': quality_results,
            'struct_path': f"{struct_dir}/{filename}.json"
        }
    
    def get_separation_confidence(self, separation_results: Dict[str, Any]) -> float:
        """Get overall confidence in separation quality."""
        if not separation_results.get('stems'):
            return 0.0
        
        confidences = [stem['quality']['confidence'] 
                      for stem in separation_results['stems'].values()]
        
        return np.mean(confidences) if confidences else 0.0
    
    def should_use_fallback(self, separation_results: Dict[str, Any], 
                           threshold: float = 0.5) -> bool:
        """Determine if fallback separation method should be used."""
        confidence = self.get_separation_confidence(separation_results)
        return confidence < threshold

