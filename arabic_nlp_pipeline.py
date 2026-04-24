import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class ArabicSentimentAnalyzer:
    """
    A Sentiment Analyzer specifically tuned for Arabic (MSA and Egyptian Dialect).
    Uses AraBERT v0.2-base as the core transformer model.
    """
    
    def __init__(self, model_name="aubmindlab/bert-base-arabertv02"):
        print(f"Initializing model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # For demonstration, we use a pre-trained sentiment model if available, 
        # or load the base for fine-tuning.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.nlp = pipeline("sentiment-analysis", model=model_name, tokenizer=self.tokenizer)
        
        # --- NEW: Expansion - Zero-Shot Emotion/Sarcasm Classification ---
        print("Initializing Zero-Shot classifier for Emotions and Sarcasm...")
        # Using xlm-roberta as it performs exceptionally well for zero-shot tasks in Arabic
        self.emotion_classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
        self.emotion_labels = ["سعادة", "غضب", "حزن", "سخرية", "شكوى"] # Happy, Angry, Sad, Sarcasm, Complaint

    def preprocess_egyptian_text(self, text):
        """
        Custom preprocessing for Egyptian Dialect and Social Media text.
        """
        # 1. Remove URLs and User Mentions
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#','', text)
        
        # 2. Character Normalization (Crucial for Arabic)
        # Normalize Alif (أ, إ, آ -> ا)
        text = re.sub("[إأآ]", "ا", text)
        # Normalize Ya (ى -> ي)
        text = re.sub("ى", "ي", text)
        # Normalize Te-Marbuta (ة -> ه)
        text = re.sub("ة", "ه", text)
        
        # 3. Remove Diacritics (Tashkeel)
        arabic_diacritics = re.compile(""" ّ | َ | ً | ُ | ٌ | ِ | ٍ | ْ | ـ """, re.VERBOSE)
        text = re.sub(arabic_diacritics, '', text)
        
        # 4. Handle Egyptian elongation (e.g., "جمييييييل" -> "جميل")
        text = re.sub(r'(.)\1+', r'\1\1', text) 
        
        return text.strip()

    def predict(self, text):
        """Predicts sentiment: Positive, Negative, or Neutral."""
        cleaned_text = self.preprocess_egyptian_text(text)
        result = self.nlp(cleaned_text)[0]
        
        emotion_result = self.emotion_classifier(cleaned_text, self.emotion_labels)
        top_emotion = emotion_result['labels'][0]
        emotion_score = emotion_result['scores'][0]

        label_map = {
            "LABEL_0": "Negative 😡",
            "LABEL_1": "Neutral 😐",
            "LABEL_2": "Positive 😊"
        }
        
        return {
            "original": text,
            "cleaned": cleaned_text,
            "sentiment": label_map.get(result['label'], result['label']),
            "confidence": f"{result['score']:.2%}",
            "top_emotion": top_emotion,
            "emotion_confidence": f"{emotion_score:.2%}"
        }

if __name__ == "__main__":
    analyzer = ArabicSentimentAnalyzer()
    
    test_tweets = [
        "الخدمة كانت زي الزفت والاكل بارد جداً",
        "بصراحة التجربة كانت عادية مفيش جديد",   
        "الله على الجمال بجد، شكراً جداً ليكم ❤️", 
        "يا حلاوة! ده انتو شركة ممتازة جدا الصراحة و خدمة العملاء بترد بعد سنة" 
    ]
    
    print("\n--- Sentiment Analysis Results ---")
    for tweet in test_tweets:
        res = analyzer.predict(tweet)
        print(f"Tweet: {res['original']}")
        print(f"Result: {res['sentiment']} (Conf: {res['confidence']})")
        print(f"Emotion/Intent: {res['top_emotion']} (Conf: {res['emotion_confidence']})\n")