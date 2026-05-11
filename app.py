import streamlit as st
import joblib

# Load Model and TF-IDF Vectorizer
model = joblib.load('fake_job_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# App Title
st.title('Fake Job Posting Detection App')

st.write('Enter job details below to check whether the job posting is Real or Fake.')

# User Inputs
title = st.text_input('Job Title')
company_profile = st.text_area('Company Profile')
description = st.text_area('Job Description')
requirements = st.text_area('Requirements')
benefits = st.text_area('Benefits')

# Combine Input Data
input_data = (
    title + ' ' +
    company_profile + ' ' +
    description + ' ' +
    requirements + ' ' +
    benefits
)

# Prediction
if st.button('Predict'):

    # Convert Text into TF-IDF
    transformed_data = tfidf.transform([input_data])

    # Predict
    prediction = model.predict(transformed_data)[0]

    # Show Result
    if prediction == 1:
        st.error('This Job Posting is FAKE')
    else:
        st.success('This Job Posting is REAL')
