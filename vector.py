import asyncio
import uuid

import httpx
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


async def get_embedding(text: str) -> list:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
        )
        return response.json()["embedding"]


async def main():
    # Initialize Qdrant client
    client = QdrantClient("localhost", port=6333)

    # Delete existing collection if it exists
    try:
        client.delete_collection("papers")
        print("Deleted existing collection")
    except Exception as e:
        print(f"No existing collection to delete: {e}")

    # Create new collection
    client.create_collection(
        collection_name="papers",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print("Created new collection")

    # Parse markdown
    soup = BeautifulSoup(markdown_content, "html.parser")
    papers = []

    # Find all table rows
    rows = soup.find("tbody").find_all("tr")

    for row in rows:
        abstract = row.get("id", "")
        if abstract == "None":
            continue

        title = row.find_all("td")[1].find("a").text

        if abstract:
            try:
                # Get embedding
                text = f"{title}\n{abstract}"
                embedding = await get_embedding(text)

                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Convert UUID to string
                    vector=embedding,
                    payload={"title": title, "abstract": abstract},
                )

                # Store in Qdrant
                client.upsert(collection_name="papers", points=[point])
                print(f"Successfully processed: {title[:50]}...")
                papers.append({"title": title, "abstract": abstract})
            except Exception as e:
                print(f"Error processing paper {title[:50]}...: {e}")

    # Test search if we have papers
    if papers:
        test_query = "What are the main approaches for discovering governing equations from data?"
        search_embedding = await get_embedding(test_query)

        # Search similar papers
        search_results = client.search(
            collection_name="papers", query_vector=search_embedding, limit=2
        )

        print("\nSearch Results:")
        for result in search_results:
            print(f"\nTitle: {result.payload['title']}")
            print(f"Score: {result.score}")


if __name__ == "__main__":
    # Put your markdown content here
    markdown_content = """... test markdown content ..."""
    asyncio.run(main())
