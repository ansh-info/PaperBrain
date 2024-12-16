import asyncio
import json
import os
import uuid
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# File to store processed file hashes
PROCESSED_FILES_LOG = "processed_files.json"


def load_processed_files():
    """Load the list of previously processed files"""
    try:
        if os.path.exists(PROCESSED_FILES_LOG):
            with open(PROCESSED_FILES_LOG, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading processed files log: {e}")
    return {}


def save_processed_files(processed_files):
    """Save the list of processed files"""
    try:
        with open(PROCESSED_FILES_LOG, "w") as f:
            json.dump(processed_files, f, indent=2)
    except Exception as e:
        print(f"Error saving processed files log: {e}")


def get_file_hash(file_path):
    """Get file hash and modification time"""
    stats = os.stat(file_path)
    return {"size": stats.st_size, "mtime": stats.st_mtime, "ctime": stats.st_ctime}


async def get_embedding(text: str) -> list:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
        )
        return response.json()["embedding"]


async def process_markdown_file(file_path: Path, client: QdrantClient) -> list:
    papers = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            markdown_content = file.read()
        print(f"\nProcessing file: {file_path.name}")

        # Parse markdown (which contains HTML)
        soup = BeautifulSoup(markdown_content, "html.parser")
        tbody = soup.find("tbody")

        if tbody is None:
            print(f"Could not find table body in {file_path.name}")
            return papers

        rows = tbody.find_all("tr")
        print(f"Found {len(rows)} papers in {file_path.name}")

        for row in rows:
            abstract = row.get("id", "")
            if abstract == "None" or not abstract:
                print("Skipping paper with no abstract")
                continue

            try:
                title = row.find_all("td")[1].find("a").text
                print(f"Processing: {title[:50]}...")

                # Get embedding
                text = f"{title}\n{abstract}"
                embedding = await get_embedding(text)

                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "title": title,
                        "abstract": abstract,
                        "source_file": file_path.name,
                    },
                )

                # Store in Qdrant
                client.upsert(collection_name="papers", points=[point])
                print(f"Successfully processed: {title[:50]}...")
                papers.append(
                    {
                        "title": title,
                        "abstract": abstract,
                        "source_file": file_path.name,
                    }
                )

            except Exception as e:
                print(
                    f"Error processing paper {title[:50] if 'title' in locals() else 'unknown'} "
                    f"from {file_path.name}: {str(e)}"
                )
                continue

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return papers


async def main():
    # Load processed files log
    processed_files = load_processed_files()

    # Directory containing markdown files
    directory = Path("markdowns")
    if not directory.exists():
        print(f"Directory {directory} does not exist!")
        return

    # Get all markdown files in the directory
    markdown_files = list(directory.glob("*.md"))
    if not markdown_files:
        print(f"No markdown files found in {directory}")
        return

    print(f"Found {len(markdown_files)} markdown files to check")

    # Initialize Qdrant client
    client = QdrantClient("localhost", port=6333)

    # Filter out already processed files that haven't changed
    files_to_process = []
    for file_path in markdown_files:
        file_hash = get_file_hash(file_path)
        if str(file_path) in processed_files:
            old_hash = processed_files[str(file_path)]
            if (
                old_hash["size"] == file_hash["size"]
                and old_hash["mtime"] == file_hash["mtime"]
            ):
                print(f"Skipping already processed file: {file_path.name}")
                continue
        files_to_process.append(file_path)

    if not files_to_process:
        print("No new or modified files to process")
        return

    print(f"Processing {len(files_to_process)} new or modified files")

    # Delete existing collection if we have files to process
    try:
        client.delete_collection("papers")
        print("Deleted existing collection")
    except Exception as e:
        print(f"No existing collection to delete: {e}")

    # Create new collection
    client.create_collection(
        collection_name="papers",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print("Created new collection")

    # Process files and collect papers
    all_papers = []
    for file_path in files_to_process:
        papers = await process_markdown_file(file_path, client)
        all_papers.extend(papers)
        # Update processed files log
        processed_files[str(file_path)] = get_file_hash(file_path)

    # Save updated processed files log
    save_processed_files(processed_files)

    print(
        f"\nProcessed {len(all_papers)} papers successfully across {len(files_to_process)} files"
    )

    # Test search if we have papers
    if all_papers:
        test_query = "What are the main approaches for discovering governing equations from data?"
        search_embedding = await get_embedding(test_query)

        # Search similar papers
        search_results = client.search(
            collection_name="papers", query_vector=search_embedding, limit=2
        )

        print("\nSearch Results:")
        for result in search_results:
            print(f"\nTitle: {result.payload['title']}")
            print(f"Source File: {result.payload['source_file']}")
            print(f"Score: {result.score}")
            print(f"Abstract preview: {result.payload['abstract'][:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
