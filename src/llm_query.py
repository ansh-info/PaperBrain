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

    async def get_llm_response(self, query: str, context: str) -> str:
        async with httpx.AsyncClient() as client:
            prompt = f"""Based on the following research papers, please answer this question: {query}

Context from relevant papers:
{context}

Please provide a comprehensive answer and cite specific papers when referring to information from them."""

            response = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt},
            )
            return response.json()["response"]

    async def search_papers(self, query: str, limit: int = 3) -> List[Dict]:
        # Get embedding for the query
        query_embedding = await self.get_embedding(query)

        # Search in Qdrant
        search_results = self.client.search(
            collection_name="papers", query_vector=query_embedding, limit=limit
        )

        return search_results

    async def search_and_respond(self, query: str, limit: int = 3):
        # Get relevant papers
        results = await self.search_papers(query, limit)

        # Build context from relevant papers
        context = "\n\n".join(
            [
                f"Title: {result.payload['title']}\nAbstract: {result.payload['abstract']}"
                for result in results
            ]
        )

        # Get LLM response
        llm_response = await self.get_llm_response(query, context)

        return results, llm_response

    def display_results(self, results, llm_response):
        print("\nAI Response:")
        print("=" * 80)
        print(llm_response)
        print("\nRelevant Papers:")
        print("=" * 80)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result.payload['title']}")
            print(f"Relevance Score: {result.score:.2f}")
            print(f"Abstract preview: {result.payload['abstract'][:200]}...")
            print("-" * 80)


async def main():
    searcher = PaperSearcher()

    while True:
        print("\nResearch Paper Q&A System")
        print("Enter your question (or 'quit' to exit):")
        query = input("> ")

        if query.lower() in ["quit", "exit", "q"]:
            break

        print("\nHow many papers to consider? (default: 3)")
        try:
            limit = int(input("> ") or "3")
        except ValueError:
            limit = 3

        try:
            results, llm_response = await searcher.search_and_respond(query, limit)
            searcher.display_results(results, llm_response)
        except Exception as e:
            print(f"Error during search: {e}")
            print("Please make sure both Qdrant and Ollama services are running")


if __name__ == "__main__":
    asyncio.run(main())
