# 📊 Brand Sentiment Analysis Pipeline

An end-to-end NLP pipeline that analyzes **75,000+ multilingual tweets and news articles** (English, Hindi, Hinglish) to determine brand-level sentiment for mobile phone brands and generate concise headlines.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.4+-green)

## 🎯 Problem Statement

In the mobile phone industry, understanding public sentiment across social media is crucial for brand strategy. The challenge is processing **multilingual data at scale** — Indian social media contains English, Hindi, and Hinglish (a mix of both) — making standard NLP tools insufficient.

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA (75K+ texts)                        │
│              Tweets + News Articles (EN/HI/Hinglish)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   STAGE 1   │
                    │ Preprocess  │  Language Detection (spaCy)
                    │ & Translate │  Script Detection (Unicode)
                    │             │  Hindi/Hinglish → English
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   STAGE 2   │
                    │   Binary    │  TF-IDF + XGBoost
                    │ Classifier  │  Mobile-tech? Yes/No
                    │             │  (Filters out 75% irrelevant data)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   STAGE 3   │
                    │    Brand    │  Regex Pattern Matching
                    │   Extract   │  100+ brands (EN + Hindi)
                    │             │  Hashtag brand detection
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼──────┐          ┌──────▼──────┐
       │   STAGE 4   │          │   STAGE 5   │
       │  Sentiment  │          │  Headline   │
       │  Analysis   │          │ Generation  │
       │  (BERT)     │          │  (T5)       │
       └──────┬──────┘          └──────┬──────┘
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │   OUTPUT    │
                    │ Brand +    │
                    │ Sentiment + │
                    │ Headline    │
                    └─────────────┘
```

## 🔬 Technical Approach

### Stage 1: Preprocessing & Translation
- **Language Detection**: spaCy + langdetect for classifying text as English, Hindi, or Hinglish
- **Script Detection**: Unicode range-based detection to differentiate Devanagari (Hindi) from Latin (Hinglish)
- **Translation**: Google Translate API with batching mechanism for API rate limits
- **Text Cleaning**: Emoji removal, URL stripping, mention removal

### Stage 2: Binary Classification (ML)
- **Vectorization**: TF-IDF with n-grams (unigram + bigram)
- **Classifier**: XGBoost with optimized hyperparameters
- **Why ML over DL?**: This step processes 4x more data on a simpler task — XGBoost achieves 95%+ accuracy while being significantly faster than transformer models
- **Evaluation**: F1-Score, Accuracy, Confusion Matrix

### Stage 3: Brand Identification
- **Approach**: Regex-based pattern matching with curated brand list (100+ brands)
- **Multilingual Support**: Hindi (Devanagari) → English brand name mapping (e.g., शाओमी → Xiaomi)
- **Hashtag Extraction**: Detects brand mentions inside hashtags
- **Position Tracking**: Records brand positions for brand-level sentiment mapping

### Stage 4: Sentiment Analysis (DL)
- **Model**: Fine-tuned BERT (`ganeshkharad/gk-hinglish-sentiment` from HuggingFace)
- **Classes**: Positive, Negative, Neutral
- **Why BERT?**: Sentiment requires understanding context and word order — "not bad" should be positive, which TF-IDF cannot capture
- **Custom Components**: PyTorch Dataset class with batch sampling for efficient GPU inference

### Stage 5: Headline Generation (DL)
- **Model**: Fine-tuned T5-base (Text-to-Text Transfer Transformer)
- **Task**: Abstractive summarization — generates new headline text (not extractive)
- **T5 vs PEGASUS**: Both were evaluated; T5 performed better on translated/noisy text
- **Beam Search**: Uses beam search (width=4) for higher quality generation

## 📁 Project Structure

```
brand-sentiment-analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── brands.py                  # Brand identification (regex + multilingual)
│   ├── utils.py                   # Preprocessing, language detection, translation
│   ├── detect_script.py           # Unicode-based script detection
│   ├── binary_classifier.py       # TF-IDF + XGBoost classifier
│   ├── sentiment_classification.py # BERT sentiment model
│   ├── sentiment_inference.py     # Sentiment prediction interface
│   └── headline_generation.py     # T5 headline generator
├── notebooks/
│   └── (experiment notebooks)
├── models/
│   └── (saved model weights)
└── data/
    └── (dataset files)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/brand-sentiment-analysis.git
cd brand-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate   # Windows

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Usage

```python
from src.binary_classifier import MobileTechClassifier
from src.sentiment_inference import SentimentClassifier
from src.headline_generation import HeadlineGenerator
from src.brands import get_brands

# Step 1: Filter mobile-tech content
classifier = MobileTechClassifier()
classifier.load()
predictions = classifier.predict(texts)

# Step 2: Extract brands
brands_found = get_brands(mobile_tech_texts)

# Step 3: Analyze sentiment
sentiment_model = SentimentClassifier(bert_path='./models/sentiment_model.pt')
sentiments = sentiment_model.predict(mobile_tech_texts)

# Step 4: Generate headlines
headline_gen = HeadlineGenerator(device='cuda', model_path='./models/headline_model.pt')
headlines = headline_gen.generate_headlines(articles)
```

## 📊 Results

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Binary Classification | TF-IDF + XGBoost | F1-Score | 0.95+ |
| Sentiment Analysis | BERT (Hinglish) | F1-Score | 0.82 |
| Headline Generation | T5-base | ROUGE-L | 0.38 |

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML Framework** | PyTorch, scikit-learn, XGBoost |
| **NLP Models** | BERT (HuggingFace), T5, XLM-RoBERTa |
| **Preprocessing** | spaCy, NLTK, syntok, demoji |
| **Translation** | Google Translate API |
| **Data Handling** | Pandas, NumPy |

## 📈 Key Design Decisions

1. **Hybrid ML/DL Architecture**: Used lightweight ML (XGBoost) where sufficient, heavy DL (BERT, T5) only where necessary — optimizing compute costs
2. **Translation-first Approach**: Translated all data to English before sentiment/headline tasks, as multilingual generation yielded poor results
3. **Regex over NER for Brand Detection**: Curated regex is faster and more reliable than NER for a fixed set of known brands

## 📝 License

This project is for educational and research purposes.
