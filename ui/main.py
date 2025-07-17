import streamlit as st
import requests
import json

API_URL = "http://api:8000/ask"

st.title("📄 Ask My Documents")
question = st.text_input("Ask a question:")
submit = st.button("Submit")

if submit and question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json={"question": question}, stream=True)
            response.raise_for_status()

            container = st.empty()
            full_text = ""
            sources_json = None

            lines = response.iter_lines(decode_unicode=True)
            for line in lines:
                if not line:
                    continue

                if line.startswith("data: "):
                    token = line.removeprefix("data: ")
                    full_text += token
                    container.markdown(f"### 🤖 Answer\n{full_text}▌")

                elif line.startswith("event: sources"):
                    try:
                        next_line = next(lines)
                        if next_line.startswith("data: "):
                            sources_json = json.loads(next_line.removeprefix("data: "))
                    except StopIteration:
                        pass

            # Final display
            container.markdown(f"### 🤖 Answer\n{full_text}")

            if sources_json:
                st.markdown("### 📚 Sources")
                for i, src in enumerate(sources_json, 1):
                    meta = src["metadata"]
                    st.markdown(f"**{i}.** `{meta.get('file_path', 'unknown')}`")
                    st.code(src["text"].strip())

        except Exception as e:
            st.error(f"Streaming failed: {e}")
