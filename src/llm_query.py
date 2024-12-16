import asyncio
import json
from typing import Dict, List

import httpx
from qdrant_client import QdrantClient


class PaperSearcher:
    def __init__(self):
        self.client = QdrantClient("localhost", port=6333)
        self.model_name = "llama3.2:1b"

    async def get_embedding(self, text: str) -> list:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
            )
            return response.json()["embedding"]

    async def get_llm_response(self, query: str, context: str) -> str:
        prompt = f"""Based on the following research papers, please answer this question: {query}

Context from relevant papers:
{context}

Please provide a comprehensive answer and cite specific papers when referring to information from them."""

        try:
            async with httpx.AsyncClient() as client:
                # First pull the model
                print(f"Ensuring {self.model_name} model is available...")
                await client.post(
                    "http://localhost:11434/api/pull",
                    json={"name": self.model_name},
                    timeout=60.0,
                )

                # Generate response
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": self.model_name, "prompt": prompt, "stream": False},
                    timeout=60.0,
                )

                if response.status_code != 200:
                    error_text = await response.text()
                    raise Exception(f"Error from Ollama API: {error_text}")

                response_data = response.json()
                if "error" in response_data:
                    raise Exception(response_data["error"])

                return response_data.get("response", "No response generated")

        except httpx.TimeoutException:
            return "Error: Request timed out. Please try again."
        except Exception as e:
            return f"Error: Unable to generate response. Details: {str(e)}"

    async def search_papers(self, query: str, limit: int = 3) -> List[Dict]:
        try:
            query_embedding = await self.get_embedding(query)
            search_results = self.client.search(
                collection_name="papers", query_vector=query_embedding, limit=limit
            )
            return search_results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    async def search_and_respond(self, query: str, limit: int = 3):
        results = await self.search_papers(query, limit)

        if not results:
            return [], "No relevant papers found."

        context = "\n\n".join(
            f"Title: {result.payload['title']}\nAbstract: {result.payload['abstract']}"
            for result in results
        )

        print(f"Generating response using {self.model_name}...")
        llm_response = await self.get_llm_response(query, context)

        return results, llm_response

    def display_results(self, results, llm_response):
        if llm_response.startswith("Error:"):
            print("\nError in AI Response:")
            print("=" * 80)
            print(llm_response)
        else:
            print("\nAI Response:")
            print("=" * 80)
            print(llm_response)

        if results:
            print("\nRelevant Papers:")
            print("=" * 80)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Title: {result.payload['title']}")
                print(f"Relevance Score: {result.score:.2f}")
                print(f"Abstract preview: {result.payload['abstract'][:200]}...")
                print("-" * 80)
        else:
            print("\nNo relevant papers found.")


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
