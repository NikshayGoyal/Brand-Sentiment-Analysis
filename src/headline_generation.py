"""
Headline Generation Module
============================
Fine-tuned T5 (Text-to-Text Transfer Transformer) for generating
concise headlines from news articles and long-form text.

Architecture:
  Long Article Text → T5 Encoder → T5 Decoder → Generated Headline

Model: t5-base (220M parameters)
Task: Abstractive summarization (generates new text, not just extraction)
"""

import os
import sys
import numpy as np
import pandas as pd
import re
import random
import pickle

import torch
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer


class HeadlineGenerator:
    """T5-based headline generator for news articles.
    
    Fine-tunes a T5-base model on article-headline pairs to generate
    concise, meaningful headlines from long-form text.
    
    Args:
        device: torch device ('cuda' or 'cpu')
        model_path: Path to fine-tuned model weights (optional)
    """
    def __init__(self, device, model_path=None):
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base').to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        
        # Load fine-tuned weights if provided
        if model_path is not None:
            self.model.load_state_dict(
                torch.load(model_path, map_location=torch.device(self.device))
            )

    def generate_batch(self, data):
        """Generate a random training batch from article-headline pairs.
        
        Args:
            data: List of (article, headline) tuples
        Returns:
            inp: List of article texts
            label: List of headline texts
        """
        output = random.sample(data, min(4, len(data)))
        inp, label = [], []
        for dat in output:
            inp.append(dat[0])
            label.append(dat[1])
        return inp, label

    def fit(self, articles, headlines, epochs=3, lr=3e-4, batch_size=4):
        """Fine-tune T5 on article-headline pairs.
        
        Args:
            articles: List of article texts
            headlines: List of corresponding headline texts
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Training batch size
        """
        assert len(articles) == len(headlines), "Articles and headlines must have same length"
        
        # Prepare training data
        data = list(zip(articles, headlines))
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = len(data) // batch_size
            
            for i in range(n_batches):
                batch_articles, batch_headlines = self.generate_batch(data)
                
                # Tokenize inputs (articles)
                input_encodings = self.tokenizer(
                    ['summarize: ' + art for art in batch_articles],
                    padding=True, truncation=True,
                    max_length=512, return_tensors='pt'
                ).to(self.device)
                
                # Tokenize targets (headlines)
                target_encodings = self.tokenizer(
                    batch_headlines, padding=True, truncation=True,
                    max_length=128, return_tensors='pt'
                ).to(self.device)
                
                labels = target_encodings.input_ids
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_encodings.input_ids,
                    attention_mask=input_encodings.attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    def generate(self, text, max_length=64, num_beams=4):
        """Generate a headline for a single article.
        
        Args:
            text: Input article text
            max_length: Maximum headline length in tokens
            num_beams: Beam search width (higher = better but slower)
        
        Returns:
            Generated headline string
        """
        self.model.eval()
        input_text = 'summarize: ' + text
        
        input_ids = self.tokenizer.encode(
            input_text, return_tensors='pt',
            max_length=512, truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        headline = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return headline

    def generate_headlines(self, texts, max_length=64, num_beams=4):
        """Generate headlines for a list of articles.
        
        Args:
            texts: List of article texts
            max_length: Maximum headline length
            num_beams: Beam search width
        
        Returns:
            List of generated headline strings
        """
        headlines = []
        for text in texts:
            headline = self.generate(text, max_length, num_beams)
            headlines.append(headline)
        return headlines

    def save_model(self, path):
        """Save model weights to disk."""
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')
