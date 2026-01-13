import re
import unicodedata


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "（": "(", "）": ")", "【": "[", "】": "]",
        "，": ",", "。": ".", "；": ";", "：": ":",
        "？": "?", "！": "!", """: '"', """: '"',
        "'": "'", "'": "'", "、": ",",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    noise_patterns = [
        r"^\s*[-·•]\s*",
        r"\s*[-·•]\s*$",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text)
    return text.strip()


def extract_aliases_from_brackets(text: str) -> tuple[str, list[str]]:
    pattern = r"[（\(]([^）\)]+)[）\)]"
    matches = re.findall(pattern, text)
    main_name = re.sub(pattern, "", text).strip()
    main_name = normalize_text(main_name)
    aliases = [normalize_text(m) for m in matches if m.strip()]
    return main_name, aliases
