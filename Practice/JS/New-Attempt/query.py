import argparse
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings  # For embeddings
from langchain.llms import GPT4All  # Free, open-source LLM
from sentence_transformers import SentenceTransformer

from get_embedding_function import get_embedding_function

FAISS_INDEX_PATH = "Practice/JS/Vectordatabase/"

PROMPT_TEMPLATE = """
You are an AI language model specializing in financial work. Answer the question based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI to accept the query text.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the FAISS DB and the embedding function
    #embedding_function = get_embedding_function()  # From your embeddings.py file
    embedding_function = SentenceTransformer('all-MiniLM-L6-v2').encode
    
    db = FAISS.load_local(FAISS_INDEX_PATH, embedding_function=embedding_function)

    # Search the FAISS index for similar chunks (top 5 results)
    results = db.similarity_search_with_score(query_text, k=5)

    # Combine context from the retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Format the prompt with context and query
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Load the free GPT4All model (you can replace this with another HuggingFace model if needed)
    model = GPT4All(model="gpt4all-lora-quantized", n_ctx=2048)  # You can change the model name to match your model file

    # Generate the response from the model
    response_text = model.invoke(prompt)

    # Extract source IDs from the retrieved documents
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    # Format and print the response with sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    return response_text


if __name__ == "__main__":
    main()
