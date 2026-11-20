import pandas as pd
import re

df = pd.read_csv("data/IMDB Dataset.csv")

df = df.drop_duplicates()

def clean_text(text):

    text = text.lower()

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text

df["cleaned_review"] = df["review"].apply(clean_text)

print("Original review:")

print(df["review"].iloc[0])

print("\nCleaned review:")

print(df["cleaned_review"].iloc[0])

df["label"] = df["sentiment"].map({

    "positive": 1,

    "negative": 0

})

# Temizleme sonrası aynı hale gelen yorumları kaldır

df = df.drop_duplicates(subset=["cleaned_review"])

# Boş kalan yorumları kaldır

df = df[df["cleaned_review"] != ""]

df.to_csv("data/cleaned_dataset.csv", index=False)

print("\nDataset successfully cleaned!")

print("Dataset shape:", df.shape)