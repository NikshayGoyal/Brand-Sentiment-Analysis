"""
Text Preprocessing & Translation Utilities
============================================
Handles multilingual text preprocessing for the sentiment analysis pipeline:
  - Language detection (English / Hindi / Hinglish)
  - Text translation (Hindi/Hinglish → English) via Google Translate API
  - Emoji removal and text cleaning
  - Sentence segmentation
  - Batch processing for API rate limits
"""

import spacy
nlp = spacy.load('en_core_web_sm')
from spacy_langdetect import LanguageDetector
from langdetect import DetectorFactory
DetectorFactory.seed = 5
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

from google_trans_new import google_translator
import time
import random
from tqdm.auto import tqdm
import demoji
if demoji.last_downloaded_timestamp() is None:
    demoji.download_codes()
import re
import syntok.segmenter as segmenter

import detect_script
import brands


def detect_lang(texts, truncate=True):
    """Detect language of each text in the input list.
    
    Uses spaCy language detection combined with Unicode script detection
    to accurately classify text as English, Hindi, or Hinglish.
    
    Args:
        texts: List of strings to detect language for
        truncate: If True, only use first 1000 chars for detection (speed optimization)
    
    Returns:
        List of language codes: 'en' (English), 'hi' (Hindi), 'hing' (Hinglish)
    """
    langs = []
    for text in tqdm(texts):
        if truncate:
            text = text[:1000]
        processed = nlp(text)
        lang = processed._.language['language']
        prob = processed._.language['score']

        if (lang != 'en' and lang != 'hi') or \
           (lang == 'en' and prob < 0.9 and detect_script.detect(text) != 'Devanagari'):
            lang = 'hing'
        elif lang == 'en' and detect_script.detect(text) == 'Devanagari':
            lang = 'hi'
        langs.append(lang)
    return langs


def _split_in_batches(article, max_len=5000):
    """Split long text into batches for API character limits.
    
    Splits at full stops to maintain sentence integrity.
    
    Args:
        article: Input text string
        max_len: Maximum characters per batch (default 5000)
    
    Returns:
        List of text chunks
    """
    batches = []
    while len(article) > max_len:
        fstop_ind = article[:max_len].rfind('.')
        if fstop_ind < 0:
            fstop_ind = max_len - 1
        batches.append(article[:fstop_ind + 1])
        article = article[fstop_ind + 1:]
    batches.append(article)
    return batches


def _translate_text(article, translator):
    """Translate a single text from Hindi to English."""
    translated_text = translator.translate(article, lang_src='hi', lang_tgt='en')
    return translated_text


def translate(texts, hinglish=False):
    """Translate a list of Hindi/Hinglish texts to English.
    
    Uses Google Translate API with automatic batching to handle
    API character limits. Includes retry logic for rate limiting.
    
    Args:
        texts: List of strings to translate
        hinglish: If True, uses 160-char limit (API constraint for Hinglish)
    
    Returns:
        translated_texts: List of English translations
        num_trans: Number of successfully translated texts
    """
    text_batches = []
    for text in texts:
        if hinglish:
            batches = _split_in_batches(text, max_len=160)
        else:
            batches = _split_in_batches(text, max_len=5000)
        text_batches.append(batches)

    translator = google_translator()
    translated_texts = []
    num_trans = 0

    for batches in tqdm(text_batches):
        translated_batch = []
        success = True
        for batch in batches:
            try:
                translated = _translate_text(batch, translator)
                translated_batch.append(translated)
                time.sleep(random.uniform(0.5, 1.5))  # Rate limiting
            except Exception as e:
                success = False
                translated_batch.append(batch)  # Keep original on failure
        
        translated_texts.append(' '.join(translated_batch))
        if success:
            num_trans += 1

    return translated_texts, num_trans


def remove_emojis(texts):
    """Remove emojis from a list of texts.
    
    Args:
        texts: List of strings
    Returns:
        List of strings with emojis removed
    """
    cleaned = []
    for text in texts:
        cleaned.append(demoji.replace(text, ''))
    return cleaned


def clean_text(text):
    """Clean a single text string by removing URLs, mentions, and special characters.
    
    Args:
        text: Input string
    Returns:
        Cleaned string
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text