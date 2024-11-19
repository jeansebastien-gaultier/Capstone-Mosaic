import streamlit as st
import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from extract_files import main as extract_files
from text_to_vectordatabase import main as process_files
from generate_response import main as generate_response
import json
import base64
from PIL import Image


def load_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e8eef8;
            border-left: 4px solid #0066cc;
            color: #333333;
        }
        .bot-message {
            background-color: #ffffff;
            border-left: 4px solid #00a3ff;
            color: #333333;
        }
        .company-info {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            color: #333333;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
            border-radius: 0.5rem;
        }
        .stTextArea>div>div>textarea {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #0066cc;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #0066cc;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            margin: 2rem 0;
        }
        .streamlit-expanderHeader {
            color: #333333;
        }
        h1, h2, h3 {
            color: #0066cc;
        }
        .logo-wrapper {
            width: 100%;
            display: flex;
            justify-content: center;
            padding: 0.5rem;  /* Reduced from 1rem */
            margin: 0 auto;
            margin-top: -3rem;  /* Added negative margin to pull everything up */
        }
        
        .logo-container {
            width: 300px;
            display: flex;
            justify-content: center;
        }
        
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            color: #0066cc;
            margin: 0.5rem 0;  /* Reduced from 2rem */
        }
        
        /* Reduce Streamlit's default padding */
        .block-container {
            padding-top: 1rem !important;  /* Reduced from default */
        }
        
        /* Center image in Streamlit */
        .stImage {
            display: flex !important;
            justify-content: center !important;
            margin-bottom: 0.5rem !important;  /* Reduced space after image */
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0066cc;
            padding: 1rem;  /* Reduced from 2rem */
            text-align: center;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.2);
            z-index: 1000;
        }
        
        .footer-text {
            color: #ffffff;
            font-size: 1.6rem;  /* Increased from 1.4rem */
            line-height: 1.5;   /* Reduced from 2 */
            margin: 0 auto;
            max-width: 100%;
        }
        
        .stSidebar {
            z-index: 1;  /* Ensure sidebar stays behind footer */
        }
        
        /* Add padding to main content to prevent footer overlap */
        .main {
            padding-bottom: 8rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def add_mosaic_branding():
    try:
        st.markdown(
            '<div class="logo-wrapper"><div class="logo-container">',
            unsafe_allow_html=True,
        )
        logo = Image.open("mosaic_logo.png")
        st.image(logo, width=600)
        st.markdown("</div></div>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(
            "Logo file not found. Please ensure mosaic_logo.png is in the project directory."
        )


# def add_mosaic_branding():
#     st.markdown("""
#         <div class="logo-wrapper">
#             <div class="logo-container">
#                 <img src="mosaic_logo.png" alt="Mosaic Logo">
#             </div>
#         </div>
#     """, unsafe_allow_html=True)


def add_footer():
    st.markdown(
        """
        <div class="footer">
            <div class="footer-text">
                <p><strong>University of Chicago Capstone Project</strong><br>
                Extracting actionable insight for investment opportunities from corporate data</p>
                <p>Created by:<br>
                Jean-Sebastien Gaultier • David Bukowski • Akshay Ramdev • Jiawei Xu</p>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "files_processed" not in st.session_state:
        st.session_state.files_processed = False
    if "current_symbol" not in st.session_state:
        st.session_state.current_symbol = None


def get_company_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "description": info.get("longBusinessSummary", "N/A"),
            "logo_url": info.get("logo_url", None),
            "website": info.get("website", "N/A"),
        }
    except:
        return None


def display_company_info(symbol):
    info = get_company_info(symbol)
    if info:
        col1, col2 = st.columns([1, 3])
        with col1:
            if info["logo_url"]:
                st.image(info["logo_url"], width=100)
            st.write(f"**{info['name']}** ({symbol})")

        with col2:
            st.write(f"**Sector:** {info['sector']}")
            st.write(f"**Industry:** {info['industry']}")
            with st.expander("Business Description"):
                st.write(info["description"])


def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        return hist
    except:
        return None


def plot_stock_chart(symbol):
    data = get_stock_data(symbol)
    if data is not None:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                )
            ]
        )
        fig.update_layout(
            title=f"{symbol} Stock Price", yaxis_title="Price", template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)


def export_chat_history():
    chat_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history": st.session_state.chat_history,
    }
    json_str = json.dumps(chat_data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="chat_history.json">Download Chat History</a>'
    st.markdown(href, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Mosaic Research Assistant", layout="wide")
    load_css()
    init_session_state()

    add_mosaic_branding()
    st.markdown(
        '<h1 class="main-title">Financial Research Assistant 📊</h1>',
        unsafe_allow_html=True,
    )

    add_footer()
    # st.title("Financial Research Assistant 📊")

    # Sidebar
    with st.sidebar:
        st.header("Company Settings")
        symbol = st.text_input("Enter Company Symbol (e.g., AAPL)").upper()

        if symbol:
            if symbol != st.session_state.current_symbol:
                st.session_state.files_processed = False
                st.session_state.current_symbol = symbol

            if st.button("Process Company Data"):
                with st.spinner("Processing company data..."):
                    process_company_data(symbol)

        if st.session_state.chat_history:
            st.header("Export Options")
            export_chat_history()

    # Main content
    if symbol:
        display_company_info(symbol)
        plot_stock_chart(symbol)

    # Chat interface
    st.header("Chat Interface 💬")

    # Display chat history
    for question, answer in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-message user-message">Q: {question}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="chat-message bot-message">A: {answer}</div>',
            unsafe_allow_html=True,
        )

    # Question input
    with st.form(key="question_form"):
        question = st.text_area("Ask about the company:", height=100)
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.form_submit_button("Send 📤")

        if submit_button:
            if not st.session_state.files_processed:
                st.error("Please process company data first!")
                return

            if not question:
                st.error("Please enter a question!")
                return

            with st.spinner("Generating response..."):
                try:
                    generate_response(symbol, question)
                    with open("response.txt", "r", encoding="utf-8") as file:
                        response = file.read()

                    st.session_state.chat_history.append((question, response))
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


def process_company_data(symbol):
    with st.spinner("Extracting company files..."):
        extract_files(symbol)

    with st.spinner("Processing and vectorizing data..."):
        process_files(symbol)

    st.session_state.files_processed = True
    st.success(f"Successfully processed data for {symbol}!")


if __name__ == "__main__":
    main()
