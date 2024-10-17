import os
import re
import string
from nltk.corpus import stopwords
from typing import List, Dict
import PyPDF2
from langchain.embeddings import HuggingFaceEmbeddings

def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and extract its text content.
    """
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.lower() # Lowercase the text
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words]) # Remove stopwords
    return text

def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split the input text into chunks of specified size with overlap.
    """
    text = preprocess_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def load_and_split_pdfs(data_folder: str) -> List[Dict]:
    """
    Load all PDFs from a folder and split them into chunks.
    """
    chunks = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(data_folder, filename)
            text = load_pdf(pdf_path)
            split_texts = split_text(text)
            chunks.extend([{"text": chunk, "source": filename} for chunk in split_texts])
    return chunks

def get_embedding_function():
    """
    Get the embedding function using HuggingFace's sentence-transformers.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdfs(data_folder: str) -> List[Dict]:
    """
    Process all PDF files in a folder: load, split into chunks, and prepare for embedding.
    """
    # Load and split the PDFs
    chunks = load_and_split_pdfs(data_folder)
    
    # Get the embedding function
    embedding_function = get_embedding_function()
    
    # Add the embedding function to each chunk for later use
    for chunk in chunks:
        chunk['embedding_function'] = embedding_function
    
    print(f"Processed {len(chunks)} chunks from PDFs in {data_folder}")
    return chunks

if __name__ == "__main__":
    # Example usage
    data_folder = "Data/"
    processed_chunks = process_pdfs(data_folder)
    print(f"Prepared {len(processed_chunks)} chunks for database insertion")