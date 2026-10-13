"""
Customer Review Sentiment Analysis Module
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) to extract
sentiment polarity scores, classifications, and key complaint keywords.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re

class CustomerSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # Add custom domain-specific telecommunication terms if needed
        self.analyzer.lexicon.update({
            'outage': -2.5,
            'throttles': -2.0,
            'buffering': -1.8,
            'unhelpful': -2.2,
            'overpriced': -2.2,
            'discrepancy': -1.5,
            'intermittent': -1.8,
            'reliable': 2.5,
            'smooth': 2.0,
            'fast': 2.0,
            'polite': 2.2
        })

    def analyze_text(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {
                "compound": 0.0,
                "pos": 0.0,
                "neu": 1.0,
                "neg": 0.0,
                "sentiment_label": "Neutral"
            }
        
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]
        
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
            
        scores["sentiment_label"] = label
        return scores

    def batch_analyze(self, reviews: list) -> pd.DataFrame:
        results = [self.analyze_text(r) for r in reviews]
        return pd.DataFrame(results)

    def extract_key_themes(self, text: str) -> list:
        """Extract key thematic tokens from customer text."""
        tokens = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'what', 'when', 'will', 'about', 'some', 'than', 'more', 'over'}
        keywords = [t for t in tokens if t not in stopwords]
        return keywords
