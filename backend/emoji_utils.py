import emoji

def convert_emoji_to_text(text):
    
    text = emoji.demojize(text)

    replacements = {
        ":face_with_tears_of_joy:": " ضحك ",
        ":red_heart:": " حب ",
        ":broken_heart:": " حزن ",
        ":angry_face:": " غضب ",
        ":smiling_face_with_heart_eyes:": " اعجاب ",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text