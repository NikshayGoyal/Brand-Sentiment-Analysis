"""
Fuzzy Brand Matching Module
============================
Enhanced brand detection using fuzzy string matching to handle:
  - Typos and misspellings (e.g., "Samsng" → "Samsung")
  - Informal abbreviations (e.g., "MI" → "Xiaomi/Mi")
  - Transliteration variations (e.g., "Oppo" vs "OPPO")

Uses rapidfuzz library for efficient fuzzy matching with 
configurable similarity thresholds.

This module extends the base regex matching in brands.py 
by adding a fuzzy matching fallback layer.
"""

from rapidfuzz import fuzz, process
from tqdm import tqdm
import re

# Import base brand list from brands module
from brands import brands as brand_set, replace_hin_to_eng, _get_brands


# Convert set to sorted list for consistent matching
BRAND_LIST = sorted([b.lower() for b in brand_set])

# Common informal brand aliases found in social media
BRAND_ALIASES = {
    'mi': 'xiaomi',
    'sammy': 'samsung',
    'ip': 'apple',
    'moto': 'motorola',
    'bb': 'blackberry',
    'op': 'oneplus',
    'noki': 'nokia',
    'poco': 'poco',
    'rm': 'realme',
}


def fuzzy_match_brand(word, threshold=80):
    """Match a single word against the brand list using fuzzy matching.
    
    Uses token_sort_ratio for comparison, which is robust to 
    word order and minor character differences.
    
    Args:
        word: Input word to match against brand list
        threshold: Minimum similarity score (0-100) to consider a match.
                   80 = allows ~1-2 character typos
                   90 = strict, only very close matches
                   
    Returns:
        Matched brand name (lowercase) or None if no match found
        
    Examples:
        fuzzy_match_brand("Samsng")   → "samsung"   (score: 91)
        fuzzy_match_brand("Xiomi")    → "xiaomi"    (score: 83)
        fuzzy_match_brand("pizza")    → None         (score: 30)
    """
    word = word.lower().strip()
    
    # Skip very short words (too many false positives)
    if len(word) < 3:
        # Check aliases for short words
        if word in BRAND_ALIASES:
            return BRAND_ALIASES[word]
        return None
    
    # Check aliases first (exact match)
    if word in BRAND_ALIASES:
        return BRAND_ALIASES[word]
    
    # Fuzzy match against brand list
    result = process.extractOne(
        word, 
        BRAND_LIST, 
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold
    )
    
    if result is not None:
        matched_brand, score, _ = result
        return matched_brand
    
    return None


def get_brands_fuzzy(texts, threshold=80, verbose=True):
    """Extract brand names from texts using regex + fuzzy matching.
    
    First attempts exact regex matching (fast), then falls back 
    to fuzzy matching for unrecognized words (catches typos).
    
    Args:
        texts: List of input text strings
        threshold: Fuzzy match similarity threshold (0-100)
        verbose: Show progress bar
        
    Returns:
        List of lists containing brand names found in each text
        
    Example:
        Input:  ["I love my Samsng phone!", "Xiomi is best"]
        Output: [["samsung"], ["xiaomi"]]
    """
    all_brands = []
    
    for text in (tqdm(texts, desc="Extracting brands") if verbose else texts):
        # Step 1: Try exact regex matching first (fast)
        exact_matches = _get_brands(text)
        
        # Step 2: Fuzzy matching on remaining words (catches typos)
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', text)
        fuzzy_matches = []
        
        for word in words:
            # Skip if already found by exact matching
            if word.lower() in exact_matches:
                continue
            
            match = fuzzy_match_brand(word, threshold=threshold)
            if match and match not in exact_matches:
                fuzzy_matches.append(match)
        
        # Combine and deduplicate
        combined = list(set(exact_matches + fuzzy_matches))
        all_brands.append(combined)
    
    return all_brands


def demonstrate_fuzzy():
    """Demo: Show fuzzy matching vs exact matching."""
    test_cases = [
        ("I love my Samsng Galaxy!", "Typo: Samsng"),
        ("Xiomi makes great phones", "Typo: Xiomi"),
        ("Appl iPhone is expensive", "Typo: Appl"),
        ("OnPlus 9 Pro is fast", "Typo: OnPlus"),
        ("Releame C55 is budget king", "Typo: Releame"),
        ("Nokia is a great brand", "Correct spelling"),
        ("I had pizza for lunch", "No brand (should return empty)"),
    ]
    
    print("=" * 70)
    print("🔍 Fuzzy Brand Matching Demo")
    print("=" * 70)
    
    for text, description in test_cases:
        # Exact matching
        exact = _get_brands(text)
        # Fuzzy matching
        fuzzy = get_brands_fuzzy([text], verbose=False)[0]
        
        improved = "✅ IMPROVED" if len(fuzzy) > len(exact) else ""
        
        print(f"\n  Text:    \"{text}\"  ({description})")
        print(f"  Regex:   {exact if exact else '❌ None found'}")
        print(f"  Fuzzy:   {fuzzy if fuzzy else '❌ None found'}  {improved}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate_fuzzy()
