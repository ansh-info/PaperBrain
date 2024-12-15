import asyncio
from typing import Dict, List

import httpx
from qdrant_client import QdrantClient


class PaperSearcher:
    def __init__(self):
        self.client = QdrantClient("localhost", port=6333)

    async def get_embedding(self, text: str) -> list:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
            )
            return response.json()["embedding"]

    async def search_papers(self, query: str, limit: int = 3) -> List[Dict]:
        # Get embedding for the query
        query_embedding = await self.get_embedding(query)

        # Search in Qdrant
        search_results = self.client.search(
            collection_name="papers", query_vector=query_embedding, limit=limit
        )

        return search_results

    def display_results(self, results):
        print("\nSearch Results:")
        print("=" * 80)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result.payload['title']}")
            print(f"Relevance Score: {result.score:.2f}")
            print(f"Abstract preview: {result.payload['abstract'][:200]}...")
            print("-" * 80)


async def main():
    searcher = PaperSearcher()

    while True:
        print("\nPaper Search Interface")
        print("Enter your query (or 'quit' to exit):")
        query = input("> ")

        if query.lower() in ["quit", "exit", "q"]:
            break

        print("\nHow many results would you like? (default: 3)")
        try:
            limit = int(input("> ") or "3")
        except ValueError:
            limit = 3

        try:
            results = await searcher.search_papers(query, limit)
            searcher.display_results(results)
        except Exception as e:
            print(f"Error during search: {e}")


if __name__ == "__main__":
    asyncio.run(main())
