import matplotlib.pyplot as plt 
import pandas as pd 


df = pd.read_csv("data/IMDB Dataset.csv")

print("\n--- İlk 5 satır ---")
print(df.head())

print("\n--- Veri setinin boyutu ---")
print(df.shape)

print("\n--- Sütun isimleri ---")
print(df.columns)

print("\n--- Veri setinin genel bilgileri ---")
df.info()

print("\n--- Eksik veriler ---")
print(df.isnull().sum())

print("\n--- Tekrarlanan veriler ---")
print(df.duplicated().sum())

print("\n--- Duygu dağılımı ---")
print(df["sentiment"].value_counts()) 

sentiment_counts = df["sentiment"].value_counts()

plt.bar(sentiment_counts.index, sentiment_counts.values)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.savefig("sentiment_distribution.png", dpi=300)
plt.show()