import re
import string
from nltk.corpus import stopwords
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import nltk
import sys

nltk.download("punkt_tab")


def get_five_sentence_chunks(text, filename):
    if "earning_call" in filename:
        sentences = text.split("\n")
        chunk_size = 3
        return [
            " ".join(sentences[i : i + chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]
    else:
        sentences = nltk.sent_tokenize(text)
        chunk_size = 5
        return [
            " ".join(sentences[i : i + chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]


def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    stop_words = set(stopwords.words("english"))
    text = " ".join(
        [word for word in text.split() if word not in stop_words]
    )  # Remove stopwords
    return text


# Embed the chunks
def embed_chunks(documents):
    text_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    for document in documents:
        document["chunk_embeddings"] = [
            {
                "text": chunk,
                "embedding": text_embedder.encode(chunk, convert_to_tensor=True),
                "raw_text": raw_chunk,
            }
            for chunk, raw_chunk in zip(document["chunks"], document["raw_chunks"])
        ]
    return documents


def main(symbol):
    documents = []
    for filename in [f"{symbol}_annual report.txt", f"{symbol}_earning_call.txt"]:
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()
        raw_chunks = get_five_sentence_chunks(text, filename)
        chunks = [preprocess_text(chunk) for chunk in raw_chunks]
        documents.append(
            {"raw_chunks": raw_chunks, "chunks": chunks, "filename": filename}
        )
    embedded_documents = embed_chunks(documents)

    embedding_dim_text = 384  # Dimension of the sentence transformer for text

    for document in embedded_documents:
        all_chunk_embeddings = []
        chunk_metadata = []
        # Create separate FAISS indices for text and image embeddings
        index_text = faiss.IndexFlatL2(embedding_dim_text)
        filename = ""
        for chunk in document["chunk_embeddings"]:
            all_chunk_embeddings.append(chunk["embedding"].cpu().numpy())
            chunk_metadata.append(
                {
                    "filename": document["filename"],
                    "text": chunk["text"],
                    "raw_text": chunk["raw_text"],
                }
            )
            filename = document["filename"]

        all_chunk_embeddings = np.array(all_chunk_embeddings)

        # Add embeddings to FAISS indices
        index_text.add(all_chunk_embeddings)
        base_filename = filename.split("_", 1)[1].rsplit(".", 1)[0]

        # Save FAISS indices and metadata
        faiss.write_index(index_text, f"{symbol}_indexes_{base_filename}.faiss")

        with open(f"{symbol}_text_metadata_{base_filename}.pkl", "wb") as f:
            pickle.dump(chunk_metadata, f)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_argument = sys.argv[1]
        main(input_argument)
    else:
        print("No input argument provided")
