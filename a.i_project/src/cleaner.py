def clean_text(text):
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = text.strip()
    return text