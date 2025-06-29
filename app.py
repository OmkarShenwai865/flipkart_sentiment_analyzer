import streamlit as st
from transformers import pipeline

# Title and subheading
st.title("🛍️ Flipkart Review Sentiment Analyzer")
st.write("Analyze customer product reviews as Positive or Negative using a pre-trained BERT model.")

# Load sentiment classifier
classifier = pipeline("sentiment-analysis")

# Text area for user input
user_input = st.text_area("Paste your product review here")

# Button and prediction
if st.button("Analyze Sentiment") and user_input.strip():
    with st.spinner("Analyzing..."):
        result = classifier(user_input)[0]
        label = result['label']
        confidence = result['score'] * 100
        st.success(f"**Sentiment:** {label}  \n**Confidence:** {confidence:.2f}%")
