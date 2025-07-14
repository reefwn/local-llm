import streamlit as st
import requests

API_URL = "http://api:8000/ask"

st.set_page_config(page_title="Ask My Docs", page_icon="📄")
st.title("📄 Ask My Documents")

question = st.text_input("Ask a question:")
submit = st.button("Submit")

if submit and question:
    with st.spinner("Thinking..."):
        try:
            res = requests.post(API_URL, json={"question": question})
            res.raise_for_status()
            data = res.json()
            st.markdown("### 🤖 Answer")
            st.success(data["answer"])

            st.markdown("### 📚 Source Chunks")
            for i, source in enumerate(data["sources"], 1):
                st.markdown(f"**{i}.** `{source['metadata'].get('file_path', 'unknown')}`")
        except Exception as e:
            st.error(f"Failed: {e}")
