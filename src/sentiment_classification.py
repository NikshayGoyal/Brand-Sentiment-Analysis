"""
Sentiment Classification Module
=================================
BERT-based sentiment classifier for multilingual (Hinglish) text.

Uses the 'ganeshkharad/gk-hinglish-sentiment' pre-trained model from 
HuggingFace, fine-tuned for 3-class sentiment classification:
  - Positive
  - Negative  
  - Neutral

Architecture:
  Input Text → Tokenizer → BERT Encoder → Classification Head → Sentiment
"""

import os
from argparse import ArgumentParser
import re

import numpy as np
import pandas as pd

import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler

import brands


class SimpleBatchDataLoader:
    """Custom batch data loader for efficient inference.
    
    Provides batched iteration over a dataset with optional shuffling.
    
    Args:
        dataset: PyTorch Dataset object
        shuffle: Whether to shuffle data (True for training, False for inference)
        drop_last: Whether to drop the last incomplete batch
        batch_size: Number of samples per batch
    """
    def __init__(self, dataset, shuffle=True, drop_last=False, batch_size=8):
        self.dataset = dataset
        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)
        self.batch_sampler = BatchSampler(
            self.sampler, drop_last=drop_last, batch_size=batch_size
        )

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for batch_idx in self.batch_sampler:
            yield self.dataset[batch_idx]


class DatasetForTokenizedSentimentClassification(torch.utils.data.Dataset):
    """Dataset for tokenized sentiment classification.
    
    Handles text tokenization using the Hinglish BERT tokenizer
    and prepares data for the sentiment classification model.
    
    Args:
        texts: List of input text strings
        args: Model hyperparameters
        idx2sentiment: Mapping from index to sentiment label
        brand2sentiment: Mapping from brand to sentiment
    """
    def __init__(self, texts, args=None, idx2sentiment=None, brand2sentiment=None):
        self.hparams = args
        self.texts = texts
        self.brand2sentiment = brand2sentiment
        self.idx2sentiment = idx2sentiment
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "ganeshkharad/gk-hinglish-sentiment"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            texts = [self.texts[i] for i in idx]
            encodings = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            )
            return encodings
        
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=512, return_tensors='pt'
        )
        return encoding


class TokenClassifier(nn.Module):
    """BERT-based token classifier for sentiment analysis.
    
    Wraps a pre-trained BERT model with a custom classification head
    that maps BERT's output to sentiment probabilities.
    
    Args:
        bert_model: Pre-trained BERT model
        threshold: Confidence threshold for classification (default 0.5)
    """
    def __init__(self, bert_model, threshold=0.5):
        super(TokenClassifier, self).__init__()
        self.bert = bert_model
        self.threshold = threshold

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

    def predict(self, input_ids, attention_mask=None):
        """Get sentiment predictions for input tokens.
        
        Returns:
            predictions: Tensor of predicted class indices
                        0 = Negative, 1 = Positive, 2 = Neutral
        """
        with torch.no_grad():
            probs = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(probs, dim=-1)
        return predictions
