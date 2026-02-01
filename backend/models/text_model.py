from transformers import pipeline
import logging

logger = logging.getLogger("WBRM_TextModel")

class TextEmotionModel:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        """
        Wrapper for HuggingFace Transformers Emotion Model.
        """
        self.pipeline = None
        self.model_name = model_name
        self.load_model()
        
    def load_model(self):
        try:
            logger.info(f"Loading Text Emotion Model: {self.model_name}")
            self.pipeline = pipeline("text-classification", model=self.model_name, top_k=None)
            logger.info("Text Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Text Model: {e}")
            self.pipeline = None

    def predict(self, text):
        """
        Returns a dictionary of emotion probabilities.
        """
        if not self.pipeline:
            logger.warning("Text model not loaded. Returning uniform distribution.")
            return None # Or default
            
        try:
            results = self.pipeline(text)
            # results: [[{'label': 'joy', 'score': 0.99}, ...]]
            
            # Map specific model labels to our standard 7 emotions if needed
            # "j-hartmann/emotion-english-distilroberta-base" outputs:
            # anger, disgust, fear, joy, neutral, sadness, surprise
            
            scores = {}
            for item in results[0]:
                scores[item['label']] = item['score']
                
            return scores
        except Exception as e:
            logger.error(f"Text inference failed: {e}")
            return None
