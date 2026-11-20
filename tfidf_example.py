import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

reviews = [
    "I love this movie",
    "I hate this movie",
    "This movie is amazing"
]

vectorizer = TfidfVectorizer()

matrix = vectorizer.fit_transform(reviews)

words = vectorizer.get_feature_names_out()

df = pd.DataFrame(matrix.toarray(), columns=words)

print(df.round(3))