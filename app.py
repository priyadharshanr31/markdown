# app.py

import streamlit as st
import tempfile
from file_utils import extract_text
from vector_store import store_chunks_in_chroma
from uuid import uuid4

st.set_page_config(page_title="Quebec Insurance Assistant", layout="wide")
st.title("📄 Upload Your Insurance Document")
st.write("Supports PDF, DOCX, and TXT files")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    st.success("File uploaded successfully ✅")

    file_bytes = uploaded_file.read()
    file_extension = uploaded_file.name.split(".")[-1].lower()

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    # Extract text
    with st.spinner("Extracting text from file..."):
        text = extract_text(tmp_path)

    if text:
        st.success("Text extraction complete.")
        st.write(text[:1000] + "..." if len(text) > 1000 else text)

        if st.button("🚀 Vectorize & Store in ChromaDB"):
            file_id = str(uuid4())

            with st.spinner("Storing vectors in ChromaDB..."):
                store_chunks_in_chroma(file_id, text)

            st.success("Chunks stored in ChromaDB successfully 🎉")
    else:
        st.error("Failed to extract text from file. Try again.")
