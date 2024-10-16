import streamlit as st
import faiss
import pickle
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path='config/.env')

# Retrieve the API token
huggingface_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

if huggingface_token is None:
    st.error("HUGGINGFACEHUB_API_TOKEN environment variable is not set.")
    st.stop()

# Initialize the LLM
llm = HuggingFaceHub(repo_id="EleutherAI/gpt-j-6B", huggingfacehub_api_token=huggingface_token)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
@st.cache_resource
def load_faiss_index():
    index_path = 'TeamBuild/financial_docs_text_index.faiss'
    if not os.path.exists(index_path):
        st.error(f"FAISS index file not found at {index_path}")
        return None
    return faiss.read_index(index_path)

# Load metadata
@st.cache_resource
def load_metadata():
    metadata_path = 'TeamBuild/financial_chunks_metadata.pkl'
    if not os.path.exists(metadata_path):
        st.error(f"Metadata file not found at {metadata_path}")
        return None
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)

# Create a custom document store class
class CustomDocstore:
    def __init__(self, metadata):
        self.metadata = metadata

    def search(self, _id):
        return self.metadata.get(_id)

# Initialize FAISS vectorstore
@st.cache_resource
def initialize_vectorstore():
    index = load_faiss_index()
    metadata = load_metadata()
    
    if index is None or metadata is None:
        return None
    
    # Initialize the custom docstore
    docstore = CustomDocstore(metadata)
    
    # Ensure you provide the index_to_docstore_id mapping
    index_to_docstore_id = {i: str(i) for i in range(len(metadata))}

    vectorstore = LangchainFAISS(
        embedding_function=embeddings.embed_query, 
        index=index, 
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore

# Function to generate answer
def generate_answer(context, question):
    prompt = f"""Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer concise.
    Context: {context}
    Question: {question}
    Helpful Answer:"""
    
    return llm(prompt)

# Streamlit UI
st.title("Financial Document Q&A System")

vectorstore = initialize_vectorstore()

if vectorstore is not None:
    query = st.text_input("Enter your question about the financial documents:")
    
    if query:
        try:
            # Retrieve relevant documents
            docs = vectorstore.similarity_search(query, k=5)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Generate answer
            result = generate_answer(context, query)
            st.write("Answer:", result)
            
            st.subheader("Sources:")
            for i, doc in enumerate(docs):
                st.write(f"Source {i+1}:")
                st.write(doc.page_content)
                st.write("Metadata:", doc.metadata)
                st.write("---")
        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")
else:
    st.error("Failed to initialize the vector store. Please check your FAISS index and metadata files.")

# Debug Information
st.subheader("Debug Information")
if vectorstore:
    st.write(f"Vectorstore type: {type(vectorstore)}")
    st.write(f"Vectorstore index type: {type(vectorstore.index)}")
    st.write(f"Number of documents in index: {vectorstore.index.ntotal}")
else:
    st.write("Vectorstore not initialized")
