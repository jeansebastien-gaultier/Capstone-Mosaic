# Extracting Actionable Intelligence for Investment Opportunities from Corporate Data

## Overview

This repository contains the work for our Capstone Project for the Master of Science program at The University of Chicago. The project, conducted in collaboration with Mosaic, aims to develop a system that helps retail investors make informed decisions by extracting and summarizing key themes from publicly available corporate data.

Our primary objective is to build a fine-tuned Large Language Model (LLM) integrated with Retrieval-Augmented Generation (RAG) to accurately answer financial queries related to investment opportunities.

## Project Team

- **Jean-Sebastien Gaultier**
- **Jiawei (Alex) Xu**
- **David Bukowski**
- **Akshay Ramdev**

### Advisors

- **Client:** Mr. Jackson Finks, Mosaic
- **Technical Advisor:** Mr. Ashish Pujari
- **Instructor:** Mr. Anil Chaturvedi

## Project Goals

### Main Goals

1. **Develop a Fine-Tuned LLM**: Create a model that accurately answers financial questions by using a fine-tuned LLM integrated with RAG, tailored to analyze corporate financial data.
2. **Data Collection and Processing**: Gather and preprocess data from sources like earnings call transcripts, investor presentations, and regulatory filings to be used in model training.
3. **Model Integration**: Integrate the fine-tuned LLM with a robust RAG component to enhance the accuracy and relevance of the responses.

### Stretch Goals

1. **Automated NLP Pipeline**: Develop an automated pipeline for data collection, cleaning, and preprocessing to continually update the model with new corporate data.
2. **Advanced Prompt Engineering**: Implement techniques for improving the quality of prompts to guide the LLM in producing more accurate and reliable responses.
3. **Prompt Classification**: Incorporate topic modeling to classify questions, directing them to the most relevant data sources and improving the model's performance.

## Data Sources

Our dataset includes:

1. **Earnings Call Transcripts**: Quarterly discussions where company executives talk about financial results, offering guidance and responding to analysts' questions.
2. **Investor Presentations**: Overviews provided by management on company performance, strategies, and market opportunities.
3. **Regulatory Filings**: Documents such as 10-K, 10-Q, and 8-K filings sourced from the SEC's EDGAR database, offering insights into a company's financial health and operational efficiency.

## Methodology

### Analytical Approach

1. **RAG Implementation**: Start with a basic RAG + LLM approach to establish a functional baseline. This will involve simple matching techniques and the use of pre-trained models like BERT.
2. **Model Fine-Tuning**: Fine-tune the LLM with the collected financial data to enhance its ability to answer domain-specific questions.
3. **Evaluation**: Measure the accuracy of the model using metrics such as cosine similarity and BERTScore to ensure the quality of both the RAG component and the LLM.

### Expected Deliverables

By the end of the project, we expect to deliver:

1. A fine-tuned LLM capable of accurately answering financial queries by summarizing complex financial documents.
2. A set of tools and scripts for data collection, preprocessing, and model fine-tuning.
3. Documentation and a detailed report on the methodology, challenges, and findings of the project.

## Usage

### Prerequisites

- Python 3.x

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/capstone-project.git
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Model

1. **Data Preprocessing**:
   - Instructions on how to preprocess the raw data for training.

2. **Model Training**:
   - Steps to train the LLM with the RAG integration.

3. **Querying the Model**:
   - How to pose financial questions to the model and interpret the results.

## Future Work

- **Integration with Real-Time Data Sources**: Expand the dataset by incorporating real-time financial data to keep the model up-to-date.
- **Performance Optimization**: Work on reducing computational costs while maintaining or improving model accuracy.
- **User Interface Development**: Create a user-friendly interface for retail investors to interact with the model.


# Running Version 1

## How to Run the Streamlit App

### Prerequisites
1. Make sure you have **Python 3.11** installed.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
1. In your terminal, navigate to the directory containing your Streamlit app files.
2. Start the app with the following command:
   ```bash
   streamlit run app.py
   ```
   > **Note**: Replace `app.py` with the name of your Streamlit application file if it’s different.

3. Open a web browser and go to `http://localhost:8501` to view the app.

### Additional Notes
- If you encounter issues, ensure that all required packages in `requirements.txt` are installed, and that your Python version is compatible.
- To stop the app, press `Ctrl + C` in your terminal.

