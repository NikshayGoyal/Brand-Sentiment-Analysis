"""
Brand Identification Module
============================
Identifies and extracts mobile phone brand names from multilingual text.
Supports:
  - English brand names
  - Hindi (Devanagari) brand names with automatic translation
  - Brand extraction from hashtags
  - Brand position tracking for brand-level sentiment mapping
"""

import re
import string
from tqdm import tqdm

# Curated set of 100+ mobile phone brand names
brands = {
    'AT&T', 'Acer', 'Allview', 'Amoi', 'Apple', 'Archos', 'Asus',
    'BQ', 'BenQ', 'BenQ-Siemens', 'Benefon', 'BlackBerry', 'Blackview',
    'Bosch', 'Casio', 'Celkon', 'Coolpad', 'Dell', 'Emporia', 'Energizer',
    'Ericsson', 'Fujitsu', 'Garmin-Asus', 'Gigabyte', 'Gionee', 'Google',
    'HP', 'HTC', 'Haier', 'Honor', 'Huawei', 'Icemobile', 'Infinix',
    'Innostream', 'Intex', 'Itel', 'Jolla', 'Karbonn', 'Kyocera',
    'LG', 'Lava', 'LeEco', 'Lenovo', 'Lepow', 'MWg', 'Maxon', 'Maxwest',
    'Meizu', 'Micromax', 'Microsoft', 'Mitac', 'Mitsubishi', 'Moto',
    'Motorola', 'Neonode', 'Nokia', 'Nvidia', 'O2', 'OnePlus', 'Oppo',
    'Orange', 'Palm', 'Panasonic', 'Pantech', 'Parla', 'Philips', 'Plum',
    'Poco', 'Prestigio', 'QMobile', 'Qtek', 'Razer', 'Realme', 'Realmi',
    'Sagem', 'Samsung', 'Sendo', 'Sewon', 'Siemens', 'Sonim', 'Sony',
    'Spice', 'T-Mobile', 'TECNO', 'Tel.Me.', 'Telit', 'Thuraya',
    'Toshiba', 'Unnecto', 'VK', 'Vertu', 'Vodafone', 'WND', 'Wiko',
    'XCute', 'XOLO', 'Xgody', 'Xiaomi', 'Redmi', 'Yezz', 'Yota',
    'ZTE', 'alcatel', 'i-mate', 'i-mobile', 'iNQ', 'verykool', 'vivo'
}

# Hindi (Devanagari) to English brand name mapping
replace_dict = {
    'एसर': 'Acer', 'एसस': 'Asus', 'बेनक्यू': 'BenQ',
    'बेनक्यू-सीमेंस': 'BenQ-Siemens', 'ब्लैकबेरी': 'Blackberry',
    'बॉश': 'Bosch', 'कैसियो': 'Casio', 'सेल्कॉन': 'Celkon',
    'कूलपैड': 'Coolpad', 'डेल': 'Dell', 'एम्पोरिया': 'Emporia',
    'एरिक्सन': 'Ericsson', 'फुजित्सु': 'Fujitsu',
    'गीगाबाइट': 'Gigabyte', 'जिओनी': 'Gionee', 'गूगल': 'Google',
    'एचपी': 'HP', 'एचटीसी': 'HTC', 'हायर': 'Haier',
    'हुवाई': 'Huawei', 'इनफिनिक्स': 'Infinix',
    'इनोस्ट्रीम': 'Innostream', 'इंटेक्स': 'Intex', 'इटेल': 'Itel',
    'कार्बन': 'Karbonn', 'एलजी': 'LG', 'लावा': 'Lava',
    'लेनोवो': 'Lenovo', 'माइक्रोमैक्स': 'Micromax',
    'माइक्रोसॉफ्ट': 'Microsoft', 'मित्सुबिशी': 'Mitsubishi',
    'मोटो': 'Moto', 'मोटोरोला': 'Motorola', 'नोकिया': 'Nokia',
    'वनप्लस': 'OnePlus', 'ओप्पो': 'Oppo', 'पैनासोनिक': 'Panasonic',
    'फिलिप्स': 'Philips', 'पोको': 'Poco', 'रेज़र': 'Razer',
    'सैमसंग': 'Samsung', 'सीमेंस': 'Siemens', 'सोनी': 'Sony',
    'टी मोबाइल': 'T-Mobile', 'तोशीबा': 'Toshiba',
    'श्याओमी': 'Xiaomi', 'आई-मोबाइल': 'i-mobile',
    'विवो': 'Vivo', 'वीवो': 'Vivo', 'रियलमी': 'Realme',
    'शाओमी': 'Xiaomi'
}

# Compile regex pattern for Hindi to English replacement
pattern = re.compile(
    r'(?<!\w)(' + '|'.join(re.escape(key) for key in replace_dict.keys()) + r')(?!\w)'
)

# Build search patterns with word boundaries
brand_list_sp = list(r'\b' + w.lower() + r'\b' for w in brands)
brand_list_sp.append(r'\b' + 'mi' + r'\b')
brand_list = list(w.lower() for w in brands)
search_exp_sp = '|'.join(brand_list_sp)
search_exp = '|'.join(brand_list)


def _find_in_hashtags(hashtags):
    """Search for brand names within hashtags."""
    matches = []
    for htag in hashtags:
        matches += re.findall(search_exp, htag, re.IGNORECASE)
    return matches


def _get_unique_brands(brandlist):
    """Remove duplicate brand mentions."""
    brandlist = [w.lower().strip() for w in brandlist]
    return list(set(brandlist))


def _get_brands(text):
    """Extract brand names from a single text string.
    
    Searches both regular text and hashtags for brand mentions.
    
    Args:
        text: Input string (tweet or article)
    Returns:
        List of unique brand names found
    """
    hashtags = re.findall(r'\B#\w*[a-zA-Z]+\w*', text)
    brands_hashtags = _find_in_hashtags(hashtags)
    brands_text = re.findall(search_exp_sp, text, re.IGNORECASE)
    brandlist = brands_hashtags + brands_text
    brandlist = _get_unique_brands(brandlist)
    return brandlist


def get_brands(texts, verbose=True):
    """Extract brand names from a list of texts.
    
    Args:
        texts: List of strings (tweets/articles)
        verbose: Show progress bar
    Returns:
        List of lists containing brand names found in each text
    """
    brandlists = []
    for text in (tqdm(texts) if verbose else texts):
        brandlist = _get_brands(text)
        brandlists.append(brandlist)
    return brandlists


def _replace_hin_to_eng(text):
    """Replace Hindi brand names with English equivalents in a single text."""
    result = pattern.sub(lambda x: replace_dict[x.group()], text)
    return result


def replace_hin_to_eng(texts):
    """Replace Hindi (Devanagari) brand names with English equivalents.
    
    Args:
        texts: List of strings
    Returns:
        List of strings with Hindi brand names replaced
    """
    repl_texts = []
    for text in texts:
        result = _replace_hin_to_eng(text)
        repl_texts.append(result)
    return repl_texts


def _get_brand_indices(text):
    """Find positions of each brand mention in a single text."""
    found_brands = _get_brands(text)
    match_indices = dict()
    for brand in found_brands:
        occ = []
        pat = r'\b' + brand + r'\b'
        split_text = re.findall(r"[\w]+|[^\s\w]", text)
        for i, word in enumerate(split_text):
            if split_text[max(0, i-1)] == '#' and re.search(brand, word, re.IGNORECASE) is not None:
                occ.append(i)
            elif re.search(pat, word, re.IGNORECASE) is not None:
                occ.append(i)
        match_indices[brand] = occ
    return match_indices


def get_brand_indices(texts):
    """Find positions of brand mentions in a list of texts.
    
    Useful for brand-level sentiment analysis — maps each brand 
    to its position in the text.
    
    Args:
        texts: List of strings
    Returns:
        List of dicts mapping brand names to their positions
        
    Example:
        'Apple and Samsung are good but Apple is better'
        => {'apple': [0, 7], 'samsung': [2]}
    """
    results = []
    for text in tqdm(texts):
        match_indices = _get_brand_indices(text)
        results.append(match_indices)
    return results
