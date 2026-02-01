import numpy as np
import time
import datetime

RECOMMENDATIONS = {
    "sad": "Consider listening to some uplifting music or talking to a friend. A short walk outside can also help.",
    "angry": "Try some deep breathing exercises. Count to ten before you speak or act. Physical activity can also be a great outlet.",
    "disgust": "Engage in an activity you find pleasant to distract yourself. A clean and fresh environment can also make a difference.",
    "fear": "Practice mindfulness or meditation to ground yourself. Remind yourself that you are in a safe space. Talk about your fears with someone you trust.",
    "happy": "That's wonderful! Share your happiness with others or channel this positive energy into a creative project.",
    "neutral": "A good time to check in with yourself. Perhaps plan your day or do a light activity you enjoy, like reading a book.",
    "surprise": "Take a moment to process this unexpected situation. Consider how you want to respond rather than reacting impulsively.",
    "default": "Take a moment to check in with your feelings. A short break, a glass of water, or some gentle stretching can be beneficial."
}

class RecommendationEngine:
    def __init__(self):
        self.recommendations_map = RECOMMENDATIONS

    def get_recommendation(self, dominant_emotion, emotion_history, time_of_day):
        """Generate personalized recommendations based on emotion history and context"""
        base_rec = self.recommendations_map.get(dominant_emotion, self.recommendations_map["default"])
        
        # Check emotion stability (variance)
        emotion_values = emotion_history.get(dominant_emotion, [])
        if len(emotion_values) >= 3:
            variance = np.var(emotion_values[-3:])
            
            # Context-specific Logic
            if dominant_emotion == "happy":
                if variance < 0.05:
                    return f"{base_rec} Your mood has been consistently positive - perfect time to tackle challenging tasks or help others."
                return f"{base_rec} Your positive energy is fluctuating - consider activities that help maintain this state."
            
            elif dominant_emotion == "sad":
                if variance < 0.05:
                    return f"{base_rec} You've been feeling down for a while. Consider a change of environment or professional support if this persists."
                return f"{base_rec} Your mood seems to be changing. Focus on self-care and activities that have lifted your spirits before."
            
            elif dominant_emotion == "angry":
                if variance < 0.05:
                     return f"{base_rec} You've been feeling frustrated for some time. Consider addressing the root cause or taking a longer break."
                return f"{base_rec} This feeling appears to be temporary. A short break might help you reset."
            
            elif dominant_emotion == "fear":
                if variance < 0.05:
                    return f"{base_rec} Your anxiety has been consistent. Consider talking to someone you trust or trying structured relaxation techniques."
                return f"{base_rec} This feeling may pass soon. Focus on what you can control in your immediate environment."
        
        # Time-of-day specific
        hour = time_of_day.hour
        if 5 <= hour < 12:
            if dominant_emotion in ["neutral", "happy"]:
                return f"{base_rec} Morning is a great time to set intentions for your day."
            return f"{base_rec} Starting your day with mindfulness might help shift your state."
        
        elif 12 <= hour < 17:
            if dominant_emotion == "neutral":
                return f"{base_rec} Afternoon is good for collaborative work."
            elif dominant_emotion in ["sad", "angry"]:
                return f"{base_rec} A short afternoon break might help refresh your perspective."
        
        elif 17 <= hour < 22:
            if dominant_emotion in ["happy", "neutral"]:
                return f"{base_rec} Evening is ideal for connecting with others."
            return f"{base_rec} Consider winding down with relaxing activities."
        
        else: # Night
            return f"{base_rec} Focus on restful activities and quality sleep."
        
        return base_rec
