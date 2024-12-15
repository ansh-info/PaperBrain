import asyncio
import uuid

import httpx
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient


async def get_embedding(text: str) -> list:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
            )
            result = response.json()
            return result["embedding"]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise


async def main():
    # Initialize Qdrant client
    client = QdrantClient("localhost", port=6333)

    # Create collection
    try:
        client.create_collection(
            collection_name="papers",
            vectors_config={
                "size": 384,  # nomic-embed-text dimension
                "distance": "Cosine",
            },
        )
    except Exception as e:
        print(f"Collection might already exist: {e}")

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
            # Get embedding
            text = f"{title}\n{abstract}"
            embedding = await get_embedding(text)

            # Generate a UUID for the point ID
            point_id = str(uuid.uuid4())

            # Store in Qdrant
            try:
                client.upsert(
                    collection_name="papers",
                    points=[
                        {
                            "id": point_id,  # Use UUID string instead of hash
                            "vector": embedding,
                            "payload": {"title": title, "abstract": abstract},
                        }
                    ],
                )
                print(f"Successfully processed: {title}")
                papers.append({"title": title, "abstract": abstract})
            except Exception as e:
                print(f"Error processing paper {title}: {e}")

    # Test search
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
    # Your markdown content here as string
    markdown_content = """... test markdown content ..."""
    asyncio.run(main())
