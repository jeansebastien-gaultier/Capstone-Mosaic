import numpy as np
import logging
from preprocessing import DocumentProcessor
from langchain_ollama import ChatOllama
from typing import List, Dict, Any
from langchain_cohere import ChatCohere
import os
from dotenv import load_dotenv
#from langchain_community.llms import Ollama

# Load environment variables from .env file
load_dotenv()

# Retrieve the token from the environment
hf_token = os.getenv("HUGGING_FACE_TOKEN")
cohere_token = os.getenv("COHERE_KEY")
logger = logging.getLogger(__name__)

class RAGModel:
    def __init__(self, processor: DocumentProcessor):
        self.processor = processor
        try:

            # self.llm = HuggingFaceEndpoint(
            #     repo_id="microsoft/Phi-3-mini-4k-instruct",
            #     task="text-generation",
            #     max_new_tokens=512,
            #     do_sample=False,
            #     repetition_penalty=1.03,
            #     huggingface_api_key=hf_token
            # )

            # self.chat = ChatHuggingFace(llm=self.llm, verbose=True)
            self.llm = ChatCohere(cohere_api_key="r7VoUmVJCmwn9jnPTp29Z9qEF32PJfH5cwI3TkDR")
        except Exception as e:
            logger.error(f"Error initializing Ollama model: {str(e)}")
            raise
    
    def get_response(self, query: str, k: int = 1) -> Dict[str, str]:
        try:
            if self.processor.index.ntotal == 0:
                return {
                    "context": "",
                    "response": "No documents have been processed yet. Please add some documents to the system first."
                }
            
            # Get query embedding
            query_embedding = self.processor.text_embedder.encode([query])
            
            # Search in FAISS index
            distances, indices = self.processor.index.search(
                query_embedding.astype(np.float32), k
            )
            
            # Get context from relevant chunks
            context = "\n".join([
                self.processor.chunk_metadata[idx]['raw_text'] 
                for idx in indices[0]
            ])
            
            # Generate response using LLM
            prompt = f"""Based on the following context, please answer the question. If the context doesn't contain relevant information, please say so.

Context: {context}

Question: {query}

Answer:"""
            
            response = self.llm.invoke(prompt)
            
            return {
                "context": context,
                "response": response.content
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "context": "",
                "response": f"An error occurred while processing your query: {str(e)}"
            }