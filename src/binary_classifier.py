"""
Binary Classifier for Mobile-Tech Content
============================================
Uses TF-IDF + XGBoost to classify whether a tweet/article 
is related to mobile technology or not.

Deliberately uses lightweight ML (not deep learning) because:
  1. This step processes 4x more data than other pipeline stages
  2. Binary classification is a simpler task
  3. XGBoost achieves 95%+ accuracy while being significantly faster

Pipeline position: Stage 2 (after preprocessing, before brand extraction)
"""

import time
import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class MobileTechClassifier:
    """Binary classifier for mobile-technology content detection.
    
    Uses TF-IDF vectorization + XGBoost to determine if a text 
    is about mobile phones/technology or not.
    
    Attributes:
        vectorizer: Trained TF-IDF vectorizer
        classifier: Trained XGBoost classifier
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2
        )
        self.classifier = XGBClassifier(
            max_depth=6,
            n_estimators=200,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.is_trained = False

    def train(self, texts, labels, test_size=0.1):
        """Train the binary classifier.
        
        Args:
            texts: List of text strings
            labels: List of binary labels (1=mobile-tech, 0=not)
            test_size: Fraction of data for validation
        
        Returns:
            Dict with training metrics (accuracy, f1_score)
        """
        t0 = time.time()

        # TF-IDF vectorization
        print("Fitting TF-IDF vectorizer...")
        X = self.vectorizer.fit_transform(texts)

        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Train XGBoost
        print("Training XGBoost classifier...")
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.classifier.predict(X_val)
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'training_time': time.time() - t0
        }

        print(f"\nTraining completed in {metrics['training_time']:.2f}s")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")

        return metrics

    def predict(self, texts):
        """Predict whether texts are mobile-tech related.
        
        Args:
            texts: List of text strings
        
        Returns:
            numpy array of predictions (1=mobile-tech, 0=not)
        """
        assert self.is_trained, "Model must be trained before prediction"
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        """Predict with probability scores.
        
        Returns:
            numpy array of shape (n_samples, 2) with probabilities
        """
        assert self.is_trained, "Model must be trained before prediction"
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def save(self, vectorizer_path='./models/tfidf_vectorizer.pkl',
             classifier_path='./models/xgb_classifier.pkl'):
        """Save trained model to disk."""
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"Model saved to {vectorizer_path} and {classifier_path}")

    def load(self, vectorizer_path='./models/tfidf_vectorizer.pkl',
             classifier_path='./models/xgb_classifier.pkl'):
        """Load trained model from disk."""
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(classifier_path, 'rb') as f:
            self.classifier = pickle.load(f)
        self.is_trained = True
        print("Model loaded successfully")
