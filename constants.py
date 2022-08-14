ARABIC_CHARS = 'دصضذطكثنتالبيسجحإأآشظمغفقةىرؤءئزوخهع'
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
NORMLIZER_MAPPER = {
    'ﻹ': 'لإ',
    'ﻷ': 'لأ',
    'ﻵ': 'لآ',
    'ﻻ': 'لا'
}
VALID_CHARS = ARABIC_CHARS + SPECIAL
KEYBOARD_KEYS = [
    'ضصثقفغعهخحجد',
    'شسيبلاتنمكط',
    'ئءؤر_ىةوزظ'
]
KEYBOARD_BLANK = '_'
