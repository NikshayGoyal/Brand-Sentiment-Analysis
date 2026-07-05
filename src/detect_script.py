"""
Script Detection Module
========================
Detects the writing script of input text using Unicode character ranges.

Supported scripts:
  - Devanagari (Hindi)
  - Bengali, Gujarati, Gurmukhi, Kannada
  - Malayalam, Oriya, Tamil, Telugu

Supported romanizations:
  - Harvard-Kyoto (HK), IAST, ITRANS, Kolkata, SLP1

Based on indic_transliteration library by Sanskrit-Coders.
License: MIT and BSD
"""

import sys
import re


class Scheme:
    """Represents a transliteration scheme."""
    Bengali = 'Bengali'
    Devanagari = 'Devanagari'
    Gujarati = 'Gujarati'
    Gurmukhi = 'Gurmukhi'
    Kannada = 'Kannada'
    Malayalam = 'Malayalam'
    Oriya = 'Oriya'
    Tamil = 'Tamil'
    Telugu = 'Telugu'
    HK = 'HK'
    IAST = 'IAST'
    ITRANS = 'ITRANS'
    Kolkata = 'Kolkata'
    SLP1 = 'SLP1'
    Velthuis = 'Velthuis'


# Unicode code point ranges for Brahmic scripts
BLOCKS = [
    ('Devanagari', 0x0900),
    ('Bengali', 0x0980),
    ('Gurmukhi', 0x0A00),
    ('Gujarati', 0x0A80),
    ('Oriya', 0x0B00),
    ('Tamil', 0x0B80),
    ('Telugu', 0x0C00),
    ('Kannada', 0x0C80),
    ('Malayalam', 0x0D00),
]

BRAHMIC_FIRST_CODE_POINT = 0x0900
BRAHMIC_LAST_CODE_POINT = 0x0D7F


class Regex:
    # Match on special Roman characters (IAST/Kolkata)
    IAST_OR_KOLKATA_ONLY = re.compile(u'[āīūṛṝḷēōṃḥñṭḍṇśṣ]')
    # Match on ITRANS-specific patterns
    ITRANS_OR_VELTHUIS_ONLY = re.compile(u'aa|ii|uu|~n')
    ITRANS_ONLY = re.compile(u'ee|oo|\\^[iI]|RR[iI]|L[iI]|'
                             u'~N|N\\^|Ch|chh|JN|sh|Sh|\\.a')
    # Match on Kolkata-specific characters
    KOLKATA_ONLY = re.compile(u'[ēō]')
    # Match on SLP1-specific characters
    SLP1_ONLY = re.compile(u'[fFxXEOCYwWqQPB]|kz|Nk|Ng|tT|dD|Sc|Sn|'
                           u'[aAiIuUfFxXeEoO]R|'
                           u'G[yr]|(\\W|^)G')
    # Match on Velthuis-specific characters
    VELTHUIS_ONLY = re.compile(u'\\.[mhnrltds]|"n|~s')


def detect(text):
    """Detect the input text's writing script.
    
    Checks each character's Unicode code point to determine
    which script block it belongs to.
    
    Args:
        text: Input string (unicode or UTF-8 encoded str)
    
    Returns:
        Script name (e.g., 'Devanagari', 'Bengali', 'Tamil')
        or romanization scheme name (e.g., 'IAST', 'ITRANS', 'HK')
    
    Example:
        detect('पितॄन्') == 'Devanagari'
        detect('hello')  == 'HK'  (default for Latin script)
    """
    if sys.version_info < (3, 0):
        try:
            text = text.decode('utf-8')
        except UnicodeError:
            pass

    # Check for Brahmic scripts using Unicode ranges
    for L in text:
        code = ord(L)
        if code >= BRAHMIC_FIRST_CODE_POINT:
            for name, start_code in BLOCKS:
                if start_code <= code <= BRAHMIC_LAST_CODE_POINT:
                    return name

    # If not Brahmic, check for romanization schemes
    if Regex.IAST_OR_KOLKATA_ONLY.search(text):
        if Regex.KOLKATA_ONLY.search(text):
            return Scheme.Kolkata
        return Scheme.IAST

    if Regex.ITRANS_OR_VELTHUIS_ONLY.search(text):
        if Regex.ITRANS_ONLY.search(text):
            return Scheme.ITRANS
        if Regex.VELTHUIS_ONLY.search(text):
            return Scheme.Velthuis

    if Regex.SLP1_ONLY.search(text):
        return Scheme.SLP1

    return Scheme.HK