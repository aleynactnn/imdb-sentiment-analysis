import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# 1. Load dataset

df = pd.read_csv("data/cleaned_dataset.csv")

X = df["cleaned_review"]
y = df["label"]


# 2. Training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Total reviews:", len(X))
print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

print("\nTraining sentiment distribution:")
print(y_train.value_counts())

print("\nTesting sentiment distribution:")
print(y_test.value_counts())


# 3. TF-IDF vectorization

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining TF-IDF shape:")
print(X_train_tfidf.shape)

print("\nTesting TF-IDF shape:")
print(X_test_tfidf.shape)

print("\nVocabulary size:")
print(len(vectorizer.get_feature_names_out()))


# 4. Create and train model

model = LogisticRegression(max_iter=1000)

print("\nTraining model...")

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# 5. Make predictions

y_pred = model.predict(X_test_tfidf)

print("\nPredictions:")
print(y_pred[:10])

print("\nActual labels:")
print(y_test.iloc[:10].values)


# 6. Model evaluation

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")


# 7. Correct and incorrect predictions

correct_predictions = (y_test == y_pred).sum()
incorrect_predictions = (y_test != y_pred).sum()

print("\nCorrect predictions:", correct_predictions)
print("Incorrect predictions:", incorrect_predictions)


# 8. Confusion matrix

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)


# 9. Classification report

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Negative", "Positive"]
))


# 10. Error analysis

for i in range(len(y_pred)):
    if y_test.iloc[i] != y_pred[i]:
        print("\nMisclassified review:")
        print(X_test.iloc[i])
        print("\nActual label:", y_test.iloc[i])
        print("Predicted label:", y_pred[i])
        break


# 11. Test custom review

review = "this movie was absolutely amazing and i loved every minute of it"

review_tfidf = vectorizer.transform([review])

prediction = model.predict(review_tfidf)

print("\nCustom review:")
print(review)

if prediction[0] == 1:
    print("Result: Positive")
else:
    print("Result: Negative")


# 12. Visualize confusion matrix

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

display.plot(cmap="Blues")

plt.title("Logistic Regression - Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=300)

# Naive Bayes model

# 13. Naive Bayes model

nb_model = MultinomialNB()

print("\nTraining Naive Bayes model...")

nb_model.fit(X_train_tfidf, y_train)

print("Naive Bayes training completed!")


# 14. Naive Bayes predictions

nb_pred = nb_model.predict(X_test_tfidf)


# 15. Naive Bayes evaluation

nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)

print("\nNaive Bayes Performance:")

print(f"Accuracy: {nb_accuracy * 100:.2f}%")
print(f"Precision: {nb_precision * 100:.2f}%")
print(f"Recall: {nb_recall * 100:.2f}%")
print(f"F1 Score: {nb_f1 * 100:.2f}%")


# 16. Model comparison

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes"],
    "Accuracy": [accuracy, nb_accuracy],
    "Precision": [precision, nb_precision],
    "Recall": [recall, nb_recall],
    "F1 Score": [f1, nb_f1]
})

print("\nModel Comparison:")
print(comparison.round(4))


# 17. Model comparison visualization

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"] * 100
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

plt.savefig("model_comparison.png", dpi=300)

plt.show()