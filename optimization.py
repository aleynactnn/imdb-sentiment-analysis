import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# 1. Load dataset

df = pd.read_csv("data/cleaned_dataset.csv")

X = df["cleaned_review"]
y = df["label"]


# 2. Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Create pipeline

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2)
    )),
    ("logistic", LogisticRegression(
        max_iter=1000
    ))
])


# 4. Hyperparameters

parameters = {
    "logistic__C": [0.1, 0.5, 1, 5, 10]
}


# 5. Grid Search

grid_search = GridSearchCV(
    pipeline,
    parameters,
    cv=3,
    scoring="accuracy",
    verbose=2,
    n_jobs=1
)


# 6. Optimization

print("\nOptimization started...")

grid_search.fit(X_train, y_train)

print("\nOptimization completed!")


# 7. Optimization results

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation accuracy:")
print(f"{grid_search.best_score_ * 100:.2f}%")


# 8. Get best model

best_model = grid_search.best_estimator_


# 9. Evaluate best model

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nOptimized Test Accuracy: {accuracy * 100:.2f}%")


# 10. Save model

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/sentiment_model.joblib")

print("\nModel successfully saved!")