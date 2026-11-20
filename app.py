import streamlit as st
import joblib

from utils import clean_text


# 1. Page configuration

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)


# 2. Load model

@st.cache_resource
def load_model():
    return joblib.load("models/sentiment_model.joblib")


model = load_model()


# 3. Sidebar

with st.sidebar:

    st.title("🎬 Sentiment AI")

    st.markdown("---")

    st.subheader("About the Project")

    st.write(
        "A machine learning application that analyzes "
        "the sentiment of English movie reviews."
    )

    st.markdown("---")

    st.subheader("Model Information")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Vectorization:** TF-IDF")
    st.write("**Dataset:** IMDb Movie Reviews")
    st.write("**Optimization:** GridSearchCV")
    st.write("**Test Accuracy:** 90.69%")

    st.markdown("---")

    st.subheader("Technologies")

    st.write("Python")
    st.write("Pandas")
    st.write("Scikit-learn")
    st.write("Streamlit")
    st.write("Natural Language Processing")


# 4. Main header

st.title("🎬 IMDb Sentiment Analysis")

st.markdown(
    "Discover the sentiment behind movie reviews "
    "with Machine Learning."
)

st.divider()


# 5. Project statistics

col1, col2, col3, col4 = st.columns(4)

col1.metric("Test Accuracy", "90.69%")
col2.metric("Original Dataset", "50,000")
col3.metric("Training Reviews", "39,660")
col4.metric("TF-IDF Features", "20,000")

st.divider()


# 6. Review input

st.subheader("📝 Analyze a Movie Review")

st.write(
    "Enter an English movie review below to discover "
    "whether it expresses a positive or negative sentiment."
)

review = st.text_area(
    "Movie Review",
    placeholder="The movie was absolutely amazing...",
    height=220
)

analyze = st.button(
    "🔍 Analyze Sentiment",
    type="primary",
    use_container_width=True
)


# 7. Prediction

if analyze:

    cleaned_review = clean_text(review)

    if cleaned_review == "":

        st.warning("Please enter a valid movie review.")

    else:

        prediction = model.predict([cleaned_review])[0]

        probabilities = model.predict_proba(
            [cleaned_review]
        )[0]

        negative_probability = probabilities[0] * 100
        positive_probability = probabilities[1] * 100

        st.divider()

        st.subheader("📊 Analysis Results")


        # 8. Sentiment result

        if prediction == 1:

            st.success("😊 POSITIVE SENTIMENT")

            st.write(
                "The reviewer expresses a positive "
                "opinion about the movie."
            )

        else:

            st.error("😞 NEGATIVE SENTIMENT")

            st.write(
                "The reviewer expresses a negative "
                "opinion about the movie."
            )


        # 9. Probability visualization

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Positive Probability",
                f"{positive_probability:.2f}%"
            )

            st.progress(
                float(positive_probability / 100)
            )

        with col2:

            st.metric(
                "Negative Probability",
                f"{negative_probability:.2f}%"
            )

            st.progress(
                float(negative_probability / 100)
            )


        # 10. Recommendation

        st.subheader("🍿 Movie Recommendation")

        if prediction == 1:

            st.info(
                "Based on this review, the movie "
                "might be worth watching!"
            )

        else:

            st.info(
                "Based on this review, you might "
                "consider another movie."
            )


# 11. Example reviews

st.divider()

with st.expander("💡 Example Movie Reviews"):

    st.markdown("**Positive Review**")

    st.write(
        "The movie was absolutely fantastic. "
        "The acting was incredible and the story "
        "kept me interested until the end."
    )

    st.markdown("**Negative Review**")

    st.write(
        "The movie was boring and disappointing. "
        "The acting was terrible and the story "
        "made no sense."
    )


# 12. Footer

st.divider()

st.caption(
    "IMDb Sentiment Analysis | Machine Learning & NLP Project"
)