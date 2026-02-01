import json
import os
from collections import Counter
from datetime import datetime

class PersonalizationEngine:
    def __init__(self, profiles_file="user_profiles.json"):
        self.profiles_file = profiles_file
        self.profiles = self.load_profiles()
        
    def load_profiles(self):
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_profiles(self):
        with open(self.profiles_file, 'w') as f:
            json.dump(self.profiles, f, indent=4)

    def update_user_emotion(self, user_id, emotion):
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                "history": [],
                "dominant_emotion": "neutral",
                "preferences": {}
            }
            
        # Add to history (keep last 100)
        timestamp = datetime.now().isoformat()
        self.profiles[user_id]["history"].append({"emotion": emotion, "time": timestamp})
        if len(self.profiles[user_id]["history"]) > 100:
            self.profiles[user_id]["history"].pop(0)
            
        # Update dominant
        emotions = [e["emotion"] for e in self.profiles[user_id]["history"]]
        if emotions:
            self.profiles[user_id]["dominant_emotion"] = Counter(emotions).most_common(1)[0][0]
            
        self.save_profiles()
        
    def get_user_profile(self, user_id):
        return self.profiles.get(user_id, None)

    def get_recommendation(self, user_id, current_emotion):
        # Basic context-aware recommendation
        # In a real system, this would use a recommender model
        
        recs = {
            "happy": ["Keep the vibe going with upbeat energetic music!", "Share your joy with a friend."],
            "sad": ["How about some calming lofi beats?", "Maybe take a short walk to clear your mind."],
            "angry": ["Deep breathing exercises might help.", "Listen to some instrumental rock to vent."],
            "fear": ["It's okay to feel this way. Try grounding techniques.", "Call a trusted friend."],
            "disgust": ["Distract yourself with a funny video.", "Drink some water and reset."],
            "surprise": ["That was unexpected! Take a moment to process.", "Capture this moment."],
            "neutral": ["A good time for productive work.", "Maybe read a book or learn something new."]
        }
        
        return recs.get(current_emotion, ["Relax and breathe."])[0]
