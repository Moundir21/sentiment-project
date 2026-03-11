"""
Arabic Text Preprocessing
=========================
هذا الملف مسؤول عن تنظيف النص العربي قبل إدخاله إلى نموذج الذكاء الاصطناعي.

المراحل:
1- تنظيف النص
2- معالجة الإيموجي
3- توحيد الحروف
4- إزالة التكرار
5- معالجة AraBERT
"""

import pandas as pd
import re
import emoji

from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar

from arabert.preprocess import ArabertPreprocessor


# تحميل معالج AraBERT
arabert_prep = ArabertPreprocessor("aubmindlab/bert-base-arabertv02")


# ----------------------------------
# إزالة الروابط
# ----------------------------------

def remove_urls(text):
    return re.sub(r"http\S+|www\S+", "", text)


# ----------------------------------
# إزالة المنشن
# ----------------------------------

def remove_mentions(text):
    return re.sub(r"@\w+", "", text)


# ----------------------------------
# إزالة الهاشتاغ
# ----------------------------------

def remove_hashtags(text):
    return re.sub(r"#\w+", "", text)


# ----------------------------------
# إزالة التشكيل
# ----------------------------------

def remove_diacritics(text):

    arabic_diacritics = re.compile("""
                                     ّ    | # Shadda
                                     َ    | # Fatha
                                     ً    | # Tanwin Fath
                                     ُ    | # Damma
                                     ٌ    | # Tanwin Damm
                                     ِ    | # Kasra
                                     ٍ    | # Tanwin Kasr
                                     ْ    | # Sukun
                                     ـ
                                 """, re.VERBOSE)

    return re.sub(arabic_diacritics, '', text)


# ----------------------------------
# إزالة تكرار الحروف
# ----------------------------------

def remove_repeated_chars(text):

    return re.sub(r'(.)\1+', r'\1\1', text)


# ----------------------------------
# تحويل الإيموجي إلى كلمات
# ----------------------------------

def convert_emojis(text):

    text = emoji.demojize(text)

    return text


# ----------------------------------
# تنظيف الرموز
# ----------------------------------

def remove_special_characters(text):

    text = re.sub(r"[^؀-ۿa-zA-Z0-9\s]", " ", text)

    return text


# ----------------------------------
# توحيد الحروف العربية
# ----------------------------------

def normalize_arabic(text):

    text = normalize_alef_ar(text)

    text = normalize_alef_maksura_ar(text)

    text = normalize_teh_marbuta_ar(text)

    return text


# ----------------------------------
# الدالة الرئيسية للتنظيف
# ----------------------------------

def clean_text(text):

    text = str(text)

    text = remove_urls(text)

    text = remove_mentions(text)

    text = remove_hashtags(text)

    text = convert_emojis(text)

    text = remove_diacritics(text)

    text = normalize_arabic(text)

    text = remove_repeated_chars(text)

    text = remove_special_characters(text)

    text = arabert_prep.preprocess(text)

    text = text.strip()

    return text


# ----------------------------------
# تحميل البيانات
# ----------------------------------

def load_data(path):

    df = pd.read_csv(path)

    # تنظيف النصوص
    df["clean_text"] = df["comment"].apply(clean_text)

    return df
