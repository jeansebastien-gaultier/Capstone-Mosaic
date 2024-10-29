import streamlit as st
import tempfile
import os
from preprocessing import DocumentProcessor
from llm import RAGModel
import logging

logger = logging.getLogger(__name__)

def initialize_session_state():
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
        st.session_state.model = RAGModel(st.session_state.processor)

def main():
    st.title("PDF Question Answering System")
    
    # Initialize session state
    initialize_session_state()
    
    # File upload section
    st.header("Upload PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Display current database status
    num_vectors = st.session_state.processor.index.ntotal
    st.info(f"Current database contains {num_vectors} text chunks from documents")
    
    if uploaded_file:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Process the PDF
            with st.spinner("Processing PDF..."):
                st.session_state.processor.process_pdf(tmp_path)
            
            st.success(f"PDF '{uploaded_file.name}' processed successfully!")
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            # Update display of database status
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
    
    # Query section
    st.header("Ask Questions")
    query = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if query:
            if st.session_state.processor.index.ntotal == 0:
                st.warning("Please upload some documents first before asking questions.")
            else:
                with st.spinner("Generating response..."):
                    try:
                        result = st.session_state.model.get_response(query)
                        
                        st.subheader("Response:")
                        st.write(result["response"])
                        
                        if result["context"]:
                            with st.expander("Show Context"):
                                st.write(result["context"])
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()