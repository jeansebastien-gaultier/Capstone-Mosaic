import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import sys


def generate_response(symbol, query, prompt_engineering):
    with open(f"{symbol}_text_metadata_annual report.pkl", "rb") as file:
        text_chunks1 = pickle.load(file)

    with open(f"{symbol}_text_metadata_earning_call.pkl", "rb") as file:
        text_chunks2 = pickle.load(file)

    client = Groq(
        api_key="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",
    )

    index1 = faiss.read_index(f"{symbol}_indexes_annual report.faiss")
    index2 = faiss.read_index(f"{symbol}_indexes_earning_call.faiss")

    text_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = text_embedder.encode([query])
    # Search in FAISS index
    distances, indices1 = index1.search(query_embedding.astype(np.float32), 1)
    distances, indices2 = index2.search(query_embedding.astype(np.float32), 1)

    context1 = "\n".join(
        [text_chunks1[idx]["raw_text"] for _, idx in enumerate(indices1[0])]
    )
    context2 = "\n".join(
        [text_chunks2[idx]["raw_text"] for _, idx in enumerate(indices2[0])]
    )

    input_text = (
        f"{prompt_engineering} \n question: {query} context: {context1} {context2}."
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_text,
            }
        ],
        model="llama3-8b-8192",
        temperature=0.0,
    )

    response = chat_completion.choices[0].message.content

    if response.startswith("Here"):
        newline_index = response.find("\n")
        if newline_index != -1:
            response = response[newline_index + 1 :].strip()

    return response


def main(symbol, question):
    # Load the pickle file

    prompt_engineering = "Answer the question using the context if helpful. Please limit the response to 5 sentences, and use memorable language that fits the tone of the company to explain the concept. No images, and no newlines just a short one sentence overall summary, then the next few building up a story to support the summary. If you can provide a better answer without the context DO NOT use the context. Please go easy on the metaphors don't make the visual language sound forced. Lastly, do NOT use 'we' - always speak in 3rd person. Don't ask if the prompt meets the requirements:"
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
