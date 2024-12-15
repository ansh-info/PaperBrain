import asyncio

import httpx
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient

# # Your markdown content that you shared
# markdown_content = """
# # Copy paste your markdown content here as a string
# """

# Or read from file
with open("markdowns/article.md", "r") as f:
    markdown_content = f.read()


async def get_embedding(text: str) -> list:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:11434/api/embeddings",  # Note: changed from embed to embeddings
                json={"model": "nomic-embed-text", "prompt": text},
            )
            result = response.json()
            print("Embedding response:", result)
            return result.get("embedding", [])
        except Exception as e:
            print(f"Error getting embedding: {e}")
            print(
                f"Response content: {response.text if 'response' in locals() else 'No response'}"
            )
            raise


async def chat_with_context(query: str, context: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": f"Context: {context}\n\nQuestion: {query}",
            },
        )
        return response.json()["response"]


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
        title = row.find_all("td")[1].find("a").text

        if abstract and abstract != "None":
            # Get embedding
            text = f"{title}\n{abstract}"
            embedding = await get_embedding(text)

            # Store in Qdrant
            client.upsert(
                collection_name="papers",
                points=[
                    {
                        "id": hash(title),
                        "vector": embedding,
                        "payload": {"title": title, "abstract": abstract},
                    }
                ],
            )
            papers.append({"title": title, "abstract": abstract})
            print(f"Processed: {title}")

    # Test search and chat
    test_query = (
        "What are the main approaches for discovering governing equations from data?"
    )
    search_embedding = await get_embedding(test_query)

    # Search similar papers
    search_results = client.search(
        collection_name="papers", query_vector=search_embedding, limit=2
    )

    # Build context from search results
    context = "\n\n".join(
        [
            f"Title: {result.payload['title']}\nAbstract: {result.payload['abstract']}"
            for result in search_results
        ]
    )

    # Chat with context
    response = await chat_with_context(test_query, context)
    print("\nQuery:", test_query)
    print("\nResponse:", response)


if __name__ == "__main__":
    asyncio.run(main())
