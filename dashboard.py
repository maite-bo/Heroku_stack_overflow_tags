import streamlit as st
import os
import requests
import json

api_endpoint = 'http://127.0.0.1:5000/tag_prediction'

def dashboard():
    st.title('Welcome to the Stack overflow tag prediction api')
    question = st.text_input("enter your question here")

    if st.button('submit'):

        data = {"question":question}

        response = requests.post(api_endpoint, json = data).json()
        result = response['tags']
        if result is not None:
            st.success('tags have been predicted')
            st.markdown(', '.join(result))

if __name__ == '__main__':
    dashboard()