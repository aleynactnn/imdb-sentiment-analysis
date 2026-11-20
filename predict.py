import joblib

# 1. Load the trained model

model = joblib.load("models/sentiment_model.joblib")


# 2. Get user input

review = input("Enter your movie review: ")

review = review.lower()


# 3. Make prediction

prediction = model.predict([review])


# 4. Display sentiment

if prediction[0] == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")


# 5. Calculate probabilities

probabilities = model.predict_proba([review])

negative_probability = probabilities[0][0] * 100
positive_probability = probabilities[0][1] * 100


# 6. Display probabilities

print(f"Negative probability: {negative_probability:.2f}%")
print(f"Positive probability: {positive_probability:.2f}%")