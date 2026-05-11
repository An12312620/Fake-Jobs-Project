import streamlit as st
import joblib
import Pipeline
import numpy as np
import pandas as pd

model = joblib.load('fake_job_model.pkl')

st.title('Fake Job Posting Detection App')

st.write('Enter job details below to check whether the job posting is Real or Fake.')

title = st.text_input('Job Title')
company_profile = st.text_area('Company Profile')
description = st.text_area('Job Description')
requirements = st.text_area('Requirements')
benefits = st.text_area('Benefits')

input_data = (
    title + ' ' +
    company_profile + ' ' +
    description + ' ' +
    requirements + ' ' +
    benefits
)

if st.button('Predict'):

    prediction = model.predict([input_data])[0]

    if prediction == 1:
        st.error('This Job Posting is FAKE')
    else:
        st.success('This Job Posting is REAL')