import pandas as pd
import pdfplumber

def load_csv(path):
    df = pd.read_csv(path)
    
    if df.empty:
        print("⚠️ CSV is empty!")
    else:
        print("✅ CSV loaded successfully")
        print(df.head())

    return df

def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text