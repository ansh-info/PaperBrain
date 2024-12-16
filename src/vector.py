import asyncio
import json
import os
import uuid
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# File to store processed papers
PROCESSED_PAPERS_LOG = "processed_papers.json"


def load_processed_papers():
    """Load the list of previously processed papers"""
    try:
        if os.path.exists(PROCESSED_PAPERS_LOG):
            with open(PROCESSED_PAPERS_LOG, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading processed papers log: {e}")
    return {}


def save_processed_papers(processed_papers):
    """Save the list of processed papers"""
    try:
        with open(PROCESSED_PAPERS_LOG, "w") as f:
            json.dump(processed_papers, f, indent=2)
    except Exception as e:
        print(f"Error saving processed papers log: {e}")


async def get_embedding(text: str) -> list:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
        )
        return response.json()["embedding"]


async def process_markdown_file(
    file_path: Path, client: QdrantClient, processed_papers: dict
) -> list:
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
        print(f"Found {len(rows)} papers to check in {file_path.name}")

        for row in rows:
            abstract = row.get("id", "")
            if abstract == "None" or not abstract:
                print("Skipping paper with no abstract")
                continue

            try:
                title = row.find_all("td")[1].find("a").text

                # Create a unique identifier for this paper
                paper_key = f"{title}_{abstract[:100]}"  # Using first 100 chars of abstract for uniqueness

                # Check if we've already processed this paper
                if paper_key in processed_papers:
                    print(f"Skipping duplicate paper: {title[:50]}...")
                    continue

                print(f"Processing new paper: {title[:50]}...")

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

                # Add to processed papers
                processed_papers[paper_key] = {
                    "title": title,
                    "source_file": file_path.name,
                    "id": point.id,
                }

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
    # Load processed papers log
    processed_papers = load_processed_papers()

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

    print(f"Found {len(markdown_files)} markdown files to process")

    # Initialize Qdrant client
    client = QdrantClient("localhost", port=6333)

    # Delete existing collection if it exists
    try:
        client.delete_collection("papers")
        print("Deleted existing collection")
        # Clear processed papers log when collection is deleted
        processed_papers = {}
    except Exception as e:
        print(f"No existing collection to delete: {e}")

    # Create new collection
    client.create_collection(
        collection_name="papers",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print("Created new collection")

    # Process all files and collect papers
    all_papers = []
    for file_path in markdown_files:
        papers = await process_markdown_file(file_path, client, processed_papers)
        all_papers.extend(papers)

    # Save processed papers log
    save_processed_papers(processed_papers)

    print(
        f"\nProcessed {len(all_papers)} unique papers successfully across {len(markdown_files)} files"
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
