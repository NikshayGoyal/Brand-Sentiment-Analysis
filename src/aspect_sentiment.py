"""
Aspect-Based Sentiment Analysis (ABSA) Module
===============================================
Extends basic sentiment analysis by breaking text into aspects
and analyzing sentiment for each aspect separately.

Instead of:
  "Camera is great but battery is bad" → Positive (incorrect overall)

We get:
  "Camera is great but battery is bad" → 
      Camera: Positive ✅
      Battery: Negative ✅

Approach:
  1. Split text at contrast conjunctions (but, however, although, etc.)
  2. Identify which aspect (feature) each segment discusses
  3. Run sentiment analysis on each segment independently
  4. Map sentiments to specific product aspects

This is a lightweight, rule-based ABSA approach suitable for 
social media text. For production, consider fine-tuning a 
dedicated ABSA model like BERT-ATE.
"""

import re
from collections import defaultdict


# Common product aspects/features for mobile phones
ASPECT_KEYWORDS = {
    'camera': ['camera', 'photo', 'photography', 'selfie', 'lens', 'megapixel', 
               'mp', 'zoom', 'portrait', 'night mode', 'video recording', 'snap'],
    'battery': ['battery', 'charge', 'charging', 'mah', 'power', 'backup', 
                'fast charge', 'wireless charge', 'battery life', 'drain'],
    'display': ['display', 'screen', 'amoled', 'oled', 'lcd', 'refresh rate',
                'hz', 'resolution', 'brightness', 'notch', 'bezel'],
    'performance': ['processor', 'chipset', 'snapdragon', 'mediatek', 'ram',
                    'speed', 'fast', 'slow', 'lag', 'hang', 'smooth', 'gaming',
                    'performance', 'benchmark', 'cpu', 'gpu'],
    'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'budget',
              'value', 'worth', 'overpriced', 'underpriced', 'money', 'rupees',
              'rs', 'inr', 'dollar'],
    'design': ['design', 'look', 'build', 'premium', 'plastic', 'glass',
               'metal', 'weight', 'thin', 'slim', 'heavy', 'color', 'colour',
               'beautiful', 'ugly', 'aesthetic'],
    'software': ['software', 'os', 'android', 'ios', 'update', 'ui', 'bloatware',
                 'oxygen', 'miui', 'one ui', 'clean', 'stock', 'app', 'feature'],
    'storage': ['storage', 'memory', 'gb', 'internal', 'expandable', 'sd card',
                'space', 'rom'],
    'audio': ['speaker', 'audio', 'sound', 'headphone', 'jack', 'dolby',
              'stereo', 'mono', 'earphone', 'music'],
    'connectivity': ['5g', '4g', 'wifi', 'bluetooth', 'nfc', 'network',
                     'signal', 'connectivity', 'sim', 'dual sim'],
}

# Contrast conjunctions that signal sentiment shifts
CONTRAST_WORDS = [
    r'\bbut\b', r'\bhowever\b', r'\balthough\b', r'\bthough\b',
    r'\byet\b', r'\bwhile\b', r'\bon the other hand\b',
    r'\bhaving said that\b', r'\bnevertheless\b', r'\bdespite\b',
    r'\bin contrast\b', r'\blekin\b', r'\bpar\b', r'\bmagar\b',  # Hindi/Hinglish
]

# Sentiment keywords for rule-based classification
POSITIVE_WORDS = {
    'amazing', 'awesome', 'excellent', 'great', 'good', 'best', 'love',
    'fantastic', 'superb', 'brilliant', 'outstanding', 'perfect', 'wonderful',
    'impressive', 'solid', 'smooth', 'fast', 'beautiful', 'premium', 'clean',
    'crisp', 'sharp', 'bright', 'long', 'quick', 'mast', 'badhiya', 'accha',
    'zabardast', 'shandar', 'killer', 'dope', 'lit', 'fire',
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'worst', 'poor', 'horrible', 'awful', 'slow',
    'disappointing', 'pathetic', 'useless', 'waste', 'ugly', 'heavy',
    'expensive', 'overpriced', 'lag', 'hang', 'drain', 'bakwas', 'bekar',
    'ghatiya', 'kharab', 'bekaar', 'worst', 'broke', 'crash', 'issue',
    'problem', 'complaint', 'hate', 'cheap', 'plastic', 'bloatware',
}


def split_at_contrasts(text):
    """Split text at contrast conjunctions.
    
    Args:
        text: Input text string
        
    Returns:
        List of text segments split at contrast words
        
    Example:
        "Camera is great but battery is bad"
        → ["Camera is great", "battery is bad"]
    """
    # Build combined pattern
    pattern = '|'.join(CONTRAST_WORDS)
    segments = re.split(pattern, text, flags=re.IGNORECASE)
    
    # Clean up segments
    segments = [s.strip() for s in segments if s.strip()]
    return segments


def detect_aspects(text):
    """Detect which product aspects are mentioned in text.
    
    Args:
        text: Input text string
        
    Returns:
        List of detected aspect names
        
    Example:
        "The camera quality is amazing" → ["camera"]
        "Battery and display are good"  → ["battery", "display"]
    """
    text_lower = text.lower()
    found_aspects = []
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_aspects.append(aspect)
                break  # One match per aspect is enough
    
    return found_aspects


def simple_sentiment(text):
    """Rule-based sentiment classification for a text segment.
    
    Counts positive and negative words to determine overall sentiment.
    
    Args:
        text: Input text string
        
    Returns:
        Tuple of (sentiment_label, confidence_score)
        sentiment_label: 'Positive', 'Negative', or 'Neutral'
        confidence_score: Float between 0 and 1
    """
    words = set(re.findall(r'\b\w+\b', text.lower()))
    
    # Check for negation (reverses sentiment)
    negation_words = {'not', 'no', 'never', 'neither', "don't", "doesn't", 
                      "isn't", "wasn't", "nahi", "na", "mat"}
    has_negation = bool(words & negation_words)
    
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)
    
    # Negation flips the sentiment
    if has_negation:
        pos_count, neg_count = neg_count, pos_count
    
    total = pos_count + neg_count
    if total == 0:
        return 'Neutral', 0.5
    
    if pos_count > neg_count:
        confidence = pos_count / total
        return 'Positive', round(confidence, 2)
    elif neg_count > pos_count:
        confidence = neg_count / total
        return 'Negative', round(confidence, 2)
    else:
        return 'Neutral', 0.5


def analyze_aspects(text):
    """Perform aspect-based sentiment analysis on a text.
    
    Main function: splits text at contrast words, detects aspects 
    in each segment, and classifies sentiment per aspect.
    
    Args:
        text: Input text string (tweet or article excerpt)
        
    Returns:
        Dict mapping aspect names to their sentiments
        
    Example:
        Input:  "Samsung camera is amazing but battery drains fast"
        Output: {
            'overall': {'sentiment': 'Positive', 'confidence': 0.75},
            'aspects': {
                'camera': {'sentiment': 'Positive', 'confidence': 1.0,
                           'text': 'Samsung camera is amazing'},
                'battery': {'sentiment': 'Negative', 'confidence': 1.0,
                            'text': 'battery drains fast'}
            }
        }
    """
    result = {
        'overall': {},
        'aspects': {}
    }
    
    # Overall sentiment
    overall_sentiment, overall_conf = simple_sentiment(text)
    result['overall'] = {
        'sentiment': overall_sentiment,
        'confidence': overall_conf
    }
    
    # Split at contrast words
    segments = split_at_contrasts(text)
    
    # If no contrast words found, analyze as single segment
    if len(segments) <= 1:
        aspects = detect_aspects(text)
        sentiment, confidence = simple_sentiment(text)
        for aspect in aspects:
            result['aspects'][aspect] = {
                'sentiment': sentiment,
                'confidence': confidence,
                'text': text.strip()
            }
        return result
    
    # Analyze each segment independently
    for segment in segments:
        aspects = detect_aspects(segment)
        sentiment, confidence = simple_sentiment(segment)
        
        for aspect in aspects:
            result['aspects'][aspect] = {
                'sentiment': sentiment,
                'confidence': confidence,
                'text': segment.strip()
            }
    
    return result


def analyze_batch(texts, verbose=True):
    """Perform aspect-based sentiment analysis on a batch of texts.
    
    Args:
        texts: List of input text strings
        verbose: Print results
        
    Returns:
        List of analysis results (one dict per text)
    """
    results = []
    for text in texts:
        result = analyze_aspects(text)
        results.append(result)
        
        if verbose:
            print(f"\n📝 \"{text}\"")
            print(f"   Overall: {result['overall']['sentiment']} "
                  f"({result['overall']['confidence']:.0%})")
            if result['aspects']:
                for aspect, info in result['aspects'].items():
                    emoji = '✅' if info['sentiment'] == 'Positive' else \
                            '❌' if info['sentiment'] == 'Negative' else '😐'
                    print(f"   {emoji} {aspect.capitalize()}: {info['sentiment']} "
                          f"({info['confidence']:.0%}) — \"{info['text']}\"")
            else:
                print("   No specific aspects detected")
    
    return results


def demonstrate_absa():
    """Demo: Show aspect-based sentiment analysis in action."""
    print("=" * 70)
    print("🎯 Aspect-Based Sentiment Analysis Demo")
    print("=" * 70)
    
    test_texts = [
        "Samsung camera is absolutely amazing but the battery drains so fast",
        "iPhone display is gorgeous however it's way too expensive for what you get",
        "OnePlus has smooth performance and great software but camera is average",
        "Redmi offers excellent battery life and affordable price",
        "Pixel camera is the best but the design looks ugly and cheap",
        "Bhai Realme ka display mast hai lekin software mein bahut bloatware hai",
    ]
    
    analyze_batch(test_texts)
    
    print("\n" + "=" * 70)
    print("✅ Notice how the SAME text can have DIFFERENT sentiments")
    print("   for different aspects — that's the power of ABSA!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_absa()
