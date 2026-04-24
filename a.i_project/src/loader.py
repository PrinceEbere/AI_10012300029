import os
import pandas as pd
import pdfplumber


# Get project root (works on Streamlit + local)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_csv(relative_path):
    """Load CSV safely from project root"""

    file_path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        print("⚠️ CSV is empty!")
    else:
        print("✅ CSV loaded successfully")
        print(df.head())

    return df


def load_pdf(relative_path):
    """Load PDF text safely from project root"""

    file_path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        print("⚠️ PDF loaded but no text found")

    return text
