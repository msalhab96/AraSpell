ARABIC_CHARS = 'دجحإﻹﻷأآﻵخهعغفقثصضذطكمنتالبيسشظزوةىﻻرؤءئ'
VALID_PUNCS = '\?\.\\\/,،\-'
NUMBERS = '0123456789'
SPECIAL = ' '
ARABIC_HARAKAT = ''.join([
    'ّ',
    'َ',
    'ً',
    'ُ',
    'ِ',
    'ٍ',
    'ْ',
    'ٌ'
])
VALID_CHARS = ARABIC_CHARS + VALID_PUNCS + NUMBERS + SPECIAL
