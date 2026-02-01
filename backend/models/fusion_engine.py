import numpy as np
import logging

logger = logging.getLogger("WBRM_Fusion")

class FusionEngine:
    def __init__(self):
        """
        Adaptive Multimodal Fusion Engine.
        Uses confidence-based weighting.
        """
        pass

    def fuse(self, vision_probs, audio_probs, text_probs, weights=None):
        """
        Fuses probabilities from 3 modalities.
        
        Args:
            vision_probs (dict): {emotion: score}
            audio_probs (dict): {emotion: score}
            text_probs (dict): {emotion: score}
            weights (dict): Optional custom weights {'vision': v, 'audio': a, 'text': t}
            
        Returns:
            dict: Fused emotion scores.
            str: Dominant emotion.
        """
        # Default Weights if not adaptive (Fallback)
        # Vision is usually most trustworthy for instant frames, Audio is secondary, Text is periodic
        if weights is None:
            # We will calculate adaptive weights based on "confidence" (max probability)
            # Higher max probability = Higher confidence = Higher weight
            
            v_conf = max(vision_probs.values()) if vision_probs else 0
            a_conf = max(audio_probs.values()) if audio_probs else 0
            t_conf = max(text_probs.values()) if text_probs else 0
            
            # Simple soft-attention mechanism
            total_conf = v_conf + a_conf + t_conf + 1e-6
            w_v = v_conf / total_conf
            w_a = a_conf / total_conf
            w_t = t_conf / total_conf
            
            # Boost text if it's very specific (explicit user input)
            if t_conf > 0.9:
                w_t *= 1.5
                
            # Renormalize
            total_w = w_v + w_a + w_t
            w_v, w_a, w_t = w_v/total_w, w_a/total_w, w_t/total_w
            
            logger.info(f"Adaptive Weights - Vision: {w_v:.2f}, Audio: {w_a:.2f}, Text: {w_t:.2f}")
        else:
            w_v = weights.get('vision', 0.33)
            w_a = weights.get('audio', 0.33)
            w_t = weights.get('text', 0.33)

        emotions = set()
        if vision_probs: emotions.update(vision_probs.keys())
        if audio_probs: emotions.update(audio_probs.keys())
        if text_probs: emotions.update(text_probs.keys())
        
        fused_scores = {e: 0.0 for e in emotions}
        
        for e in emotions:
            val_v = vision_probs.get(e, 0)
            val_a = audio_probs.get(e, 0)
            val_t = text_probs.get(e, 0)
            
            # Weighted Sum Fusion
            fused_scores[e] = (val_v * w_v) + (val_a * w_a) + (val_t * w_t)
            
        dominant_emotion = max(fused_scores, key=fused_scores.get)
        return fused_scores, dominant_emotion
