import os
import json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def initialize_vectorstore():

    jsonl_path = "./assignment_chat/a_z_wall_mounting_1.jsonl"
    persist_directory = "./assignment_chat/chroma_db_"

    # Create embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # If ChromaDB already exists, load it
    if os.path.exists(persist_directory):
        print("Loading existing vector store...")

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="a_z"
        )

    # Otherwise, create a new ChromaDB
    else:
        print("Creating new vector store...")

        documents = []

        with open(jsonl_path, "r", encoding="utf-8") as file:

            for line in file:

                if not line.strip():
                    continue

                item = json.loads(line)

                # Get text
                content = item["body"]["input"]

                # Get chunk ID
                custom_id = item.get("custom_id", "")

                # Split custom_id into metadata
                parts = custom_id.split("_")

                page = parts[0]
                seq_num = parts[1]
                start_index = parts[2]

                # Store text AND metadata in the Document
                document = Document(
                    page_content=content,
                    metadata={
                        "custom_id": custom_id,
                        "page": page,
                        "seq_num": seq_num,
                        "start_index": start_index
                    }
                )

                documents.append(document)

        print("Embedding chunks...")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="a_z",
            persist_directory=persist_directory
        )

        print("Vector store created!")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever