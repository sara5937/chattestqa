import streamlit as st
from MEDI_PDF_CHATBOT.data_ingestion import Data_loading
from MEDI_PDF_CHATBOT.embedding import download_embedding
from MEDI_PDF_CHATBOT.model_api import load_model
from MEDI_PDF_CHATBOT.retriver import redriever_model

# Initialize the Streamlit app
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 MEDI PDF Chatbot")
st.sidebar.header("Upload and Chat")

# Upload PDF file
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Load the PDF file
        documents = Data_loading(uploaded_file)

        # Load models
        llm, embeddings = load_model()

        # Generate embeddings and upload to Qdrant
        vector_store = download_embedding(embeddings, documents)

        st.success("PDF successfully processed and stored in the database!")

        # Chat interface
        st.write("## Chat with Your Document")

        # User input for chat
        user_input = st.text_input("Ask a question about your document:", "")

        if user_input:
            with st.spinner("Generating response..."):
                # Retrieve and generate the response
                response = redriever_model(vector_store, llm, user_input, chain_type_kwargs={})
                st.write("### Response:")
                st.write(response)
