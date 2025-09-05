import re

EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
ACCOUNT = re.compile(r'\b\d{10,16}\b')  # naive example
PHONE = re.compile(r'\b\+?\d[\d\s\-()]{7,}\b')

def redact(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = ACCOUNT.sub("[ACCT]", text)
    text = PHONE.sub("[PHONE]", text)
    return text
