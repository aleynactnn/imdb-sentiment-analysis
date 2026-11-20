# IMDb Sentiment Analysis

A Machine Learning and Natural Language Processing project
that classifies English movie reviews as positive or negative
using TF-IDF and Logistic Regression.

## Project Overview

The purpose of this project is to develop a sentiment
classification model and integrate it into an interactive
web application using Streamlit.

## Project Statistics

| Metric | Value |
|---|---|
| Original Dataset | 50,000 reviews |
| Cleaned Dataset | 49,576 reviews |
| Training Samples | 39,660 |
| Testing Samples | 9,916 |
| TF-IDF Features | 20,000 |
| Optimized Test Accuracy | 90.69% |

## Technologies Used

- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib
- Natural Language Processing

## Machine Learning Workflow

1. Data collection
2. Exploratory data analysis
3. Text preprocessing
4. Duplicate removal
5. Train-test split
6. TF-IDF feature extraction
7. Model training
8. Model comparison
9. Hyperparameter optimization
10. Model evaluation
11. Model serialization
12. Web application development

## Text Preprocessing

The preprocessing pipeline includes:

- Lowercase conversion
- HTML tag removal
- URL removal
- Number removal
- Punctuation removal
- Whitespace normalization

## Model Selection

Logistic Regression and Multinomial Naive Bayes were
evaluated during model development.

Logistic Regression was selected as the final model
because of its better classification performance.

GridSearchCV was used to optimize the model.

## Results

The optimized Logistic Regression model achieved
a test accuracy of 90.69%.

### Sentiment Distribution

![Sentiment Distribution](sentiment_distribution.png)

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### Model Comparison

![Model Comparison](model_comparison.png)

## Application Features

- English movie review analysis
- Positive and negative sentiment classification
- Sentiment probability scores
- Interactive Streamlit interface
- Review-based movie recommendations

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

## Command Line Prediction

```bash
python predict.py
```

## Dataset

IMDb Dataset of 50K Movie Reviews:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

The dataset is not included in this repository.

To reproduce model training, download the dataset and
place it inside the data folder.

Then run:

```bash
python preprocessing.py
python optimization.py
```

## Limitations

The model is trained on English movie reviews.
Sarcasm, ambiguous expressions and reviews from other
domains may reduce prediction performance.

## Future Improvements

- Neutral sentiment classification
- Multilingual sentiment analysis
- Deep learning models
- Rating prediction

## Conclusion

This project demonstrates an end-to-end machine learning
workflow, from raw text data to an interactive web
application.