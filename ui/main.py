import streamlit as st
import requests

API_URL = "http://api:8000/ask"

st.title("📄 Ask My Documents")
question = st.text_input("Ask a question:")
submit = st.button("Submit")

if submit and question:
    with st.spinner("Streaming response..."):
        try:
            response = requests.post(API_URL, json={"question": question}, stream=True)
            response.raise_for_status()

            container = st.empty()
            full_text = ""

            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    token = line.removeprefix("data: ")
                    full_text += token
                    container.markdown(f"### 🤖 Answer\n{full_text}▌")

            container.markdown(f"### 🤖 Answer\n{full_text}")

        except Exception as e:
            st.error(f"Failed to stream: {e}")
