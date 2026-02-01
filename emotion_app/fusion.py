import numpy as np

EMOTIONS = ["sad", "angry", "disgust", "fear", "happy", "neutral", "surprise"]

class EmotionFusion:
    def __init__(self, alpha=0.7):
        """
        Args:
            alpha (float): Smoothing factor (0 < alpha < 1). 
                           Higher alpha = more weight to previous value (smoother, more lag).
                           Lower alpha = more weight to current value (responsive, more jitter).
        """
        self.alpha = alpha
        self.smoothed_emotions = {e: 0.0 for e in EMOTIONS}
        self.smoothed_emotions["neutral"] = 1.0 # Start neutral

    def fuse(self, visual_scores, audio_scores, text_scores, 
             visual_conf=1.0, audio_conf=1.0, text_conf=1.0):
        """
        Multimodal Fusion via Weighted Confidence.
        Equation: E_final = sum(w * c * E) / sum(w * c)
        """
        final_scores = {e: 0.0 for e in EMOTIONS}
        
        # Weights (Base importance of modality)
        # Visual is usually most reliable for specific expressions
        # Text is very high confidence if present
        # Audio is supportive
        w_v = 1.0
        w_t = 1.5 if text_conf > 0.5 else 1.0 
        w_a = 0.8
        
        total_weight = 0.0
        
        # Fusion Accumulation
        for e in EMOTIONS:
            # Weighted contributions
            v_contrib = w_v * visual_conf * visual_scores.get(e, 0.0)
            t_contrib = w_t * text_conf * text_scores.get(e, 0.0)
            a_contrib = w_a * audio_conf * audio_scores.get(e, 0.0)
            
            final_scores[e] = v_contrib + t_contrib + a_contrib
            
        # Normalize
        total_val = sum(final_scores.values())
        if total_val > 0:
            for e in EMOTIONS:
                final_scores[e] /= total_val
        else:
            final_scores["neutral"] = 1.0 # Default
            
        # Exponential Smoothing
        # smoothed[t] = alpha * smoothed[t-1] + (1 - alpha) * current[t]
        for e in EMOTIONS:
            self.smoothed_emotions[e] = (self.alpha * self.smoothed_emotions[e]) + \
                                        ((1 - self.alpha) * final_scores[e])
            
        # Re-normalize to ensure sum is exactly 1.0 after smoothing
        total_smooth = sum(self.smoothed_emotions.values())
        if total_smooth > 0:
            for e in EMOTIONS:
                self.smoothed_emotions[e] /= total_smooth
                
        return self.smoothed_emotions

    def calculate_stability_score(self, emotion_history):
        """
        Stability Score = 1 / (1 + Variance)
        Higher score = More stable.
        """
        # We look at the variance of the dominant emotion over recent history
        if not emotion_history:
            return 1.0
            
        # Flatten history to get variance of 'happiness' or 'sadness' over time
        # For simplicity, we can just look at the variance of the most frequent dominant emotion
        # But here, let's just return a placeholder based on recent consistency if available
        return 0.8 # Placeholder if not fully calculating temporal variance here
