"""
Sentiment Inference Module
===========================
Loads a trained BERT sentiment model and runs inference on new text.

Usage:
    classifier = SentimentClassifier(bert_path='./models/sentiment_model.pt')
    sentiments = classifier.predict(texts)
"""

import os
import pickle
from glob import glob
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import transformers

import brands
import utils
from sentiment_classification import (
    DatasetForTokenizedSentimentClassification,
    SimpleBatchDataLoader,
    TokenClassifier
)


class SentimentClassifier:
    """End-to-end sentiment classifier for multilingual text.
    
    Loads a fine-tuned BERT model and provides an interface
    for sentiment prediction on new text data.
    
    Args:
        bert_path: Path to saved model weights
        threshold: Confidence threshold for classification
    """
    def __init__(self, bert_path, threshold=0.5, **kwargs):
        # Load pre-trained Hinglish BERT
        gk_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            'ganeshkharad/gk-hinglish-sentiment', num_labels=3
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Wrap with custom classifier
        self.model = TokenClassifier(gk_model, threshold=threshold)
        self.model.load_state_dict(torch.load(bert_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "ganeshkharad/gk-hinglish-sentiment"
        )
        del gk_model  # Free memory

        # Sentiment label mapping
        self.idx2sentiment = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}

    def predict(self, texts, batch_size=16):
        """Predict sentiment for a list of texts.
        
        Args:
            texts: List of input strings
            batch_size: Batch size for inference
            
        Returns:
            List of sentiment labels ('Positive', 'Negative', 'Neutral')
        """
        dataset = DatasetForTokenizedSentimentClassification(texts)
        dataloader = SimpleBatchDataLoader(
            dataset, shuffle=False, batch_size=batch_size
        )
        
        all_predictions = []
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(self.device)
            attention_mask = batch['attention_mask'].squeeze(1).to(self.device)
            
            predictions = self.model.predict(input_ids, attention_mask)
            all_predictions.extend(predictions.cpu().numpy())
        
        # Map indices to labels
        sentiments = [self.idx2sentiment[pred] for pred in all_predictions]
        return sentiments

    def predict_with_scores(self, texts, batch_size=16):
        """Predict sentiment with confidence scores.
        
        Args:
            texts: List of input strings
            batch_size: Batch size for inference
            
        Returns:
            List of dicts with sentiment label and confidence score
        """
        dataset = DatasetForTokenizedSentimentClassification(texts)
        dataloader = SimpleBatchDataLoader(
            dataset, shuffle=False, batch_size=batch_size
        )
        
        results = []
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(self.device)
            attention_mask = batch['attention_mask'].squeeze(1).to(self.device)
            
            with torch.no_grad():
                probs = self.model(input_ids, attention_mask)
                preds = torch.argmax(probs, dim=-1)
                scores = torch.max(probs, dim=-1).values
            
            for pred, score in zip(preds.cpu().numpy(), scores.cpu().numpy()):
                results.append({
                    'sentiment': self.idx2sentiment[pred],
                    'confidence': float(score)
                })
        
        return results
