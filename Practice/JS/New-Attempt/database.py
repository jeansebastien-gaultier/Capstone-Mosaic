import os
import shutil
import argparse
from langchain.vectorstores import Chroma
from get_embedding_function import get_embedding_function  # from your embeddings.py

CHROMA_PATH = "chroma"
DATA_PATH = "data"  # Assuming embeddings or processed chunks are saved here

def main():
    # Check if the database should be reset
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing the database...")
        clear_database()

    # Load preprocessed chunks (from embeddings.py or another source)
    chunks = load_preprocessed_chunks()

    # Add chunks to ChromaDB
    populate_chroma(chunks)

def load_preprocessed_chunks():
    print(f"📂 Loading preprocessed chunks from {DATA_PATH}")
    
    # Assuming you have a function to load already embedded chunks from embeddings.py
    from embeddings import load_embedded_chunks
    chunks = load_embedded_chunks(DATA_PATH)
    
    print(f"✅ Loaded {len(chunks)} chunks")
    return chunks

def populate_chroma(chunks):
    print(f"🚀 Connecting to ChromaDB at {CHROMA_PATH}")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Calculate chunk IDs (if not already done during preprocessing)
    chunks = calculate_chunk_ids(chunks)

    # Load existing items from the database
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"🔍 Number of existing documents: {len(existing_ids)}")

    # Add new chunks that aren't in the DB
    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]
    if new_chunks:
        print(f"📥 Adding {len(new_chunks)} new chunks to the database")
        db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
        db.persist()
    else:
        print("✅ No new chunks to add")

def calculate_chunk_ids(chunks):
    print("🔢 Calculating chunk IDs")
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"🗑️ Database at {CHROMA_PATH} cleared")
    else:
        print(f"❌ No database found at {CHROMA_PATH}")

if __name__ == "__main__":
    main()
