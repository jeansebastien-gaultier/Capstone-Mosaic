import streamlit as st
import os
from extract_files import main as extract_files
from text_to_vectordatabase import main as process_files
from generate_response import main as generate_response

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'files_processed' not in st.session_state:
        st.session_state.files_processed = False

def process_company_data(symbol):
    with st.spinner('Extracting company files...'):
        extract_files(symbol)
    
    with st.spinner('Processing and vectorizing data...'):
        process_files(symbol)
    
    st.session_state.files_processed = True
    st.success(f'Successfully processed data for {symbol}')

def main():
    st.set_page_config(page_title="Company Research Assistant", layout="wide")
    init_session_state()
    
    st.title("Company Research Assistant")
    
    # Sidebar for company input
    with st.sidebar:
        st.header("Settings")
        symbol = st.text_input("Enter Company Symbol (e.g., AAPL)").upper()
        if st.button("Process Company Data"):
            if symbol:
                process_company_data(symbol)
            else:
                st.error("Please enter a company symbol")
    
    # Main chat interface
    st.header("Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for question, answer in st.session_state.chat_history:
            st.text_area("Question", value=question, height=50, disabled=True)
            st.text_area("Answer", value=answer, height=100, disabled=True)
            st.markdown("---")
    
    # Input for new questions
    with st.form(key='question_form'):
        question = st.text_area("Ask a question about the company:", height=100)
        submit_button = st.form_submit_button("Send")
        
        if submit_button:
            if not st.session_state.files_processed:
                st.error("Please process company data first!")
                return
                
            if not question:
                st.error("Please enter a question!")
                return
            
            with st.spinner('Generating response...'):
                try:
                    # Save response to file and read it back
                    generate_response(symbol, question)
                    with open("response.txt", "r", encoding="utf-8") as file:
                        response = file.read()
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, response))
                    st.experimental_rerun()
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()