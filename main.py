import os
import shutil
import streamlit as st

from core import user_input, get_pdf_text, get_text_chunks, get_vector_store

# Delete existing local files (only once at the beginning)
if "delete_done" not in st.session_state:
    def delete_local(folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    delete_local("faiss_index")
    st.session_state["delete_done"] = True

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDFs")
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf"])

        client = st.text_input("Enter your Google API Key", type="password")
        if client:
            st.session_state["GOOGLE_API_KEY"] = client

        if st.button("Submit & Process"):
            
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        st.info("This app allows you to chat with PDF multiple files")

    if not client or not pdf_docs:
        st.markdown("<h4>Instructions<", unsafe_allow_html=True)

        st.write("1. Upload your PDF files")
        st.write("2. Ask a question")
        st.write("3. Get an answer")

    else:
        user_question = st.text_input("Ask a Question from the PDF Files")
        if st.button("Ask"):
            st.write(user_input(user_question))

if __name__ == "__main__":
    main()