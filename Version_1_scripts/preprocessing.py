import re
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
import os
import faiss
import pickle
from typing import List, Dict, Any
import logging

print(np.__version__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, vector_db_path: str = 'Vectordatabase'):
        """
        Initialize the DocumentProcessor with a vector database path.
        Creates necessary directories and files if they don't exist.
        """
        self.vector_db_path = vector_db_path
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        # Create vector database directory if it doesn't exist
        os.makedirs(self.vector_db_path, exist_ok=True)
        logger.info(f"Using vector database path: {self.vector_db_path}")
        
        # Initialize paths
        self.index_path = os.path.join(self.vector_db_path, 'document_index.faiss')
        self.metadata_path = os.path.join(self.vector_db_path, 'document_metadata.pkl')
        
        # Initialize or load index and metadata
        self.initialize_storage()

    def initialize_storage(self):
        """Initialize or load the FAISS index and metadata"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                logger.info("Loading existing index and metadata...")
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
            else:
                logger.info("Creating new index and metadata...")
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.chunk_metadata = []
                # Save empty index and metadata
                self.save_storage()
            
            logger.info(f"Index contains {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error initializing storage: {str(e)}")
            # Reinitialize with empty storage
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.chunk_metadata = []
            self.save_storage()

    def save_storage(self):
        """Save the current index and metadata"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
            logger.info("Successfully saved index and metadata")
        except Exception as e:
            logger.error(f"Error saving storage: {str(e)}")
            raise

    @staticmethod
    def get_full_text(pdf_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
        return text

    # @staticmethod
    # def get_five_sentence_chunks(text: str) -> List[str]:
    #     sentences = nltk.tokenize.sent_tokenize(text)
    #     chunk_size = 5
    #     return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    
    @staticmethod
    def get_five_sentence_chunks(text):
        """
        Splits text into chunks of approximately 5 sentences each.
        Args:
            text (str): The input text to be split into chunks.
        Returns:
            List[str]: List of strings, where each string contains approximately 5 sentences.
        """
        # Basic regex to split sentences by punctuation followed by whitespace and an uppercase letter
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = [s.strip() for s in re.split(sentence_endings, text) if s.strip()]

        # Group sentences into chunks of 5
        chunk_size = 5
        chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
        return chunks

    @staticmethod
    def preprocess_text(text: str) -> str:
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        stop_words = set(stopwords.words('english'))
        text = " ".join([word for word in text.split() if word not in stop_words])
        return text

    def process_pdf(self, pdf_path: str) -> None:
        """Process a single PDF and add it to the vector database"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            text = self.get_full_text(pdf_path)
            raw_chunks = self.get_five_sentence_chunks(text)
            chunks = [self.preprocess_text(chunk) for chunk in raw_chunks]
            
            # Create embeddings
            embeddings = []
            filename = os.path.basename(pdf_path)
            
            for chunk, raw_chunk in zip(chunks, raw_chunks):
                embedding = self.text_embedder.encode(chunk)
                embeddings.append(embedding)
                self.chunk_metadata.append({
                    'filename': filename,
                    'text': chunk,
                    'raw_text': raw_chunk
                })
            
            if embeddings:  # Only add if we have embeddings
                # Add to FAISS index
                self.index.add(np.array(embeddings))
                # Save updated index and metadata
                self.save_storage()
                logger.info(f"Successfully processed {filename}")
            else:
                logger.warning(f"No embeddings generated for {filename}")
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise