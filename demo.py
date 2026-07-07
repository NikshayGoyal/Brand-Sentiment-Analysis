"""
Brand Sentiment Analysis Pipeline — Demo Script
==================================================
Demonstrates the complete pipeline flow:
  Raw Text → Preprocessing → Classification → Brand Extraction → Sentiment → Headlines

Usage:
    python demo.py
"""

from src.brands import get_brands, replace_hin_to_eng
from src.detect_script import detect


def demo_script_detection():
    """Demo: Detect writing script of text."""
    print("=" * 60)
    print("STAGE 0: Script Detection")
    print("=" * 60)

    test_cases = [
        ("Samsung is a great brand", "English (Latin)"),
        ("शाओमी का नया फोन अच्छा है", "Hindi (Devanagari)"),
        ("Bhai phone mast hai", "Hinglish (Latin)"),
    ]

    for text, expected in test_cases:
        script = detect(text)
        print(f"  Input:    '{text}'")
        print(f"  Script:   {script}")
        print(f"  Expected: {expected}")
        print()


def demo_brand_extraction():
    """Demo: Extract brand names from multilingual text."""
    print("=" * 60)
    print("STAGE 3: Brand Identification")
    print("=" * 60)

    sample_texts = [
        "I love my new Samsung Galaxy S21! The camera is amazing #apple",
        "शाओमी का नया Redmi Note बहुत अच्छा है",
        "OnePlus 9 Pro vs iPhone 13 — which is better? #nokia",
        "Bhai Realme ka phone bahut sasta hai, Vivo se accha",
    ]

    # First, replace Hindi brand names with English equivalents
    print("\n📝 Hindi → English Brand Name Replacement:")
    replaced = replace_hin_to_eng(sample_texts)
    for original, cleaned in zip(sample_texts, replaced):
        if original != cleaned:
            print(f"  Before: '{original}'")
            print(f"  After:  '{cleaned}'")
            print()

    # Extract brands from text
    print("🏷️  Brand Extraction Results:")
    brands_found = get_brands(replaced, verbose=False)
    for text, brands in zip(sample_texts, brands_found):
        print(f"  Text:   '{text[:60]}...'")
        print(f"  Brands: {brands}")
        print()


def demo_pipeline_overview():
    """Demo: Show the complete pipeline flow."""
    print("=" * 60)
    print("📊 COMPLETE PIPELINE OVERVIEW")
    print("=" * 60)
    print("""
    ┌─────────────────────────────────────────┐
    │  RAW DATA (75K+ multilingual texts)     │
    └──────────────────┬──────────────────────┘
                       │
                ┌──────▼──────┐
                │  STAGE 1    │  spaCy + Unicode Detection
                │ Preprocess  │  Google Translate API
                │ & Translate │  Emoji/URL removal
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │  STAGE 2    │  TF-IDF + XGBoost
                │  Binary     │  "Is this about mobile tech?"
                │ Classifier  │  Filters 75% irrelevant data
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │  STAGE 3    │  Regex (100+ brands)
                │   Brand     │  Hindi→English mapping
                │  Extract    │  Hashtag detection
                └──────┬──────┘
                       │
          ┌────────────┴────────────┐
          │                         │
   ┌──────▼──────┐          ┌──────▼──────┐
   │  STAGE 4    │          │  STAGE 5    │
   │ Sentiment   │          │ Headline    │
   │   (BERT)    │          │   (T5)      │
   └──────┬──────┘          └──────┬──────┘
          │                         │
          └────────────┬────────────┘
                       │
                ┌──────▼──────┐
                │   OUTPUT    │
                │ Brand +     │
                │ Sentiment + │
                │ Headline    │
                └─────────────┘
    """)

    # Simulate pipeline output
    print("📋 Sample Output:")
    print("-" * 60)
    sample_output = [
        {
            "text": "Samsung Galaxy S21 has an amazing camera but battery is average",
            "brand": "Samsung",
            "sentiment": "Positive",
            "confidence": 0.87,
            "headline": "Samsung Galaxy S21 impresses with camera quality"
        },
        {
            "text": "iPhone 13 price is too high for what it offers",
            "brand": "Apple",
            "sentiment": "Negative",
            "confidence": 0.82,
            "headline": "iPhone 13 faces criticism over pricing"
        },
        {
            "text": "OnePlus launched new phone with Snapdragon 888",
            "brand": "OnePlus",
            "sentiment": "Neutral",
            "confidence": 0.91,
            "headline": "OnePlus unveils new Snapdragon 888 smartphone"
        },
    ]

    for item in sample_output:
        print(f"  Brand:      {item['brand']}")
        print(f"  Sentiment:  {item['sentiment']} ({item['confidence']:.0%} confidence)")
        print(f"  Headline:   \"{item['headline']}\"")
        print(f"  Source:     \"{item['text'][:50]}...\"")
        print()


if __name__ == "__main__":
    print("\n🚀 Brand Sentiment Analysis Pipeline — Demo\n")
    
    demo_pipeline_overview()
    demo_script_detection()
    demo_brand_extraction()

    print("=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
    print("\nNote: Sentiment (BERT) and Headline (T5) stages require")
    print("trained model weights. See README.md for full setup instructions.")
