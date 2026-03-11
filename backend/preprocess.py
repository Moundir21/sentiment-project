import pandas as pd
import re
import emoji

from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar

from arabert.preprocess import ArabertPreprocessor


arabert_prep = ArabertPreprocessor("aubmindlab/bert-base-arabertv02")


def remove_urls(text):
    return re.sub(r"http\S+|www\S+", "", text)


def remove_mentions(text):
    return re.sub(r"@\w+", "", text)


def remove_hashtags(text):
    return re.sub(r"#\w+", "", text)


def remove_diacritics(text):

    arabic_diacritics = re.compile("""
        ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ
    """, re.VERBOSE)

    return re.sub(arabic_diacritics, '', text)


def remove_repeated_chars(text):

    return re.sub(r'(.)\1+', r'\1\1', text)


def convert_emojis(text):

    return emoji.demojize(text, language='en')


def remove_special_characters(text):

    return re.sub(r"[^؀-ۿa-zA-Z0-9\s:]", " ", text)


def normalize_arabic(text):

    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)

    return text


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


def load_data(path):

    df = pd.read_csv(path)

    df = df.dropna()

    df["clean_text"] = df["comment"].apply(clean_text)

    return df
