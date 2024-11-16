import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import sys


def generate_response(symbol, query, prompt_engineering):
    with open(f"{symbol}_text_metadata.pkl", "rb") as file:
        text_chunks = pickle.load(file)

    client = Groq(
        api_key="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",
    )

    index = faiss.read_index(f"{symbol}_indexes.faiss")

    text_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = text_embedder.encode([query])
    # Search in FAISS index
    distances, indices = index.search(query_embedding.astype(np.float32), 1)

    context = "\n".join(
        [text_chunks[idx]["raw_text"] for _, idx in enumerate(indices[0])]
    )

    input_text = f"question: {query} context: {context}. {prompt_engineering}"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_text,
            }
        ],
        model="llama3-8b-8192",
    )

    response = chat_completion.choices[0].message.content

    if response.startswith("Here"):
        newline_index = response.find("\n")
        if newline_index != -1:
            response = response[newline_index + 1 :].strip()

    return response


def main(symbol, question):
    # Load the pickle file

    prompt_engineering = "Please limit the response to 5 sentences, and use visual language to explain the concept. No images, and no newlines just a short one sentence overall summary, then the next few building up a story to support the summary."
    response = generate_response(symbol, question, prompt_engineering)
    with open(f"response.txt", "w", encoding="utf-8") as file:
        file.write(response)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        symbol = sys.argv[1]
        question = sys.argv[2]
        main(symbol, question)
    else:
        print("No input argument provided")
