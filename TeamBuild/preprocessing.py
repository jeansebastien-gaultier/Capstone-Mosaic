import re
import string
from nltk.corpus import stopwords
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import pdfplumber
import os
import faiss
import pickle

# Preprocessing function
def preprocess_text(text):
    text = text.lower() # Lowercase the text
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words]) # Remove stopwords
    return text

# Initialize tokenizer for token-based chunking
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Function to chunk text based on tokens
def chunk_text_by_tokens(text, max_tokens=512):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True)) 
    return chunks

# Initialize text and image embedding models
text_embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Text embeddings

# Function to extract and embed images from PDFs
def extract_text_and_images_with_preprocessing(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text from page
            page_text = page.extract_text()
            if page_text:
                text += page_text  # Append to overall text

    # Preprocess and chunk the text
    text = preprocess_text(text)
    chunks = chunk_text_by_tokens(text, max_tokens=512)

    return chunks

def process_pdfs_with_chunking_and_image_embedding(data_folder):
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            chunks = extract_text_and_images_with_preprocessing(pdf_path)
            documents.append({
                'chunks': chunks,
                'filename': filename
            })
    return documents

# Folder containing your PDFs
data_folder = '../Data/'

# Process the PDFs with chunking and image embedding
documents = process_pdfs_with_chunking_and_image_embedding(data_folder)

# Embed the chunks
def embed_chunks(documents):
    for document in documents:
        document['chunk_embeddings'] = [{'text': chunk, 'embedding': text_embedder.encode(chunk, convert_to_tensor=True)} for chunk in document['chunks']]
    return documents

# Embed the chunks of all PDFs
embedded_documents = embed_chunks(documents)


build_folder = '../Vectordatabase/'
if not os.path.exists(build_folder):
    os.makedirs(build_folder)

# Initialize FAISS index for text chunks and image embeddings
embedding_dim_text = 384  # Dimension of the sentence transformer for text

# Create separate FAISS indices for text and image embeddings
index_text = faiss.IndexFlatL2(embedding_dim_text)

# Flatten all chunk embeddings and image embeddings and store their metadata
all_chunk_embeddings = []
chunk_metadata = []

for document in embedded_documents:
    # Add text chunk embeddings to index
    for chunk in document['chunk_embeddings']:
        all_chunk_embeddings.append(chunk['embedding'].cpu().numpy())
        chunk_metadata.append({'filename': document['filename'], 'text': chunk['text']})
    

# Convert embeddings to numpy arrays
all_chunk_embeddings = np.array(all_chunk_embeddings)

# Add embeddings to FAISS indices
index_text.add(all_chunk_embeddings)

# Save FAISS indices and metadata
faiss.write_index(index_text, 'financial_docs_text_index.faiss')


with open('financial_chunks_metadata.pkl', 'wb') as f:
    pickle.dump(chunk_metadata, f)

