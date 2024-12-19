import asyncio
import json
import textwrap
from collections import Counter
from datetime import datetime
from typing import Dict, List

import httpx
from qdrant_client import QdrantClient


class PaperSearcher:
    def __init__(self):
        self.client = QdrantClient("localhost", port=6333)
        self.search_history = []
        self.shown_papers = set()  # Track papers shown in current session

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

        # Search with higher limit to account for potential duplicates
        search_limit = limit * 2
        search_results = self.client.search(
            collection_name="papers", query_vector=query_embedding, limit=search_limit
        )

        # Filter out previously shown papers
        unique_results = []
        for result in search_results:
            paper_id = result.payload["title"]  # Using title as unique identifier
            if paper_id not in self.shown_papers:
                self.shown_papers.add(paper_id)
                unique_results.append(result)
                if len(unique_results) >= limit:
                    break

        # Log the search
        self.log_search(query, unique_results)

        return unique_results

    def log_search(self, query: str, results: List[Dict]):
        """Log search query and results for analysis"""
        self.search_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "num_results": len(results),
                "avg_relevance": (
                    sum(r.score for r in results) / len(results) if results else 0
                ),
            }
        )

    def get_search_analytics(self):
        """Analyze search patterns"""
        if not self.search_history:
            return "No searches performed yet."

        total_searches = len(self.search_history)
        avg_relevance = (
            sum(h["avg_relevance"] for h in self.search_history) / total_searches
        )

        return {
            "total_searches": total_searches,
            "average_relevance": avg_relevance,
            "unique_papers_shown": len(self.shown_papers),
        }

    def explain_relevance_score(self, score: float) -> str:
        """Explain the relevance score in human terms"""
        if score >= 0.9:
            return "Extremely relevant - Very strong match with query"
        elif score >= 0.8:
            return "Highly relevant - Strong match with query"
        elif score >= 0.7:
            return "Moderately relevant - Good match with query"
        elif score >= 0.6:
            return "Somewhat relevant - Partial match with query"
        else:
            return "Less relevant - Weak match with query"

    def display_results(self, results, show_analytics=False):
        print("\nSearch Results:")
        print("=" * 80)

        if not results:
            print("No matching papers found.")
            return

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result.payload['title']}")

            # Display and explain relevance score
            relevance_score = result.score
            print(f"Relevance Score: {relevance_score:.2f}")
            print(
                f"Relevance Explanation: {self.explain_relevance_score(relevance_score)}"
            )

            # Format abstract with proper wrapping
            abstract_preview = textwrap.fill(
                (
                    result.payload["abstract"][:300] + "..."
                    if len(result.payload["abstract"]) > 300
                    else result.payload["abstract"]
                ),
                width=80,
                initial_indent="",
                subsequent_indent="    ",
            )
            print(f"\nAbstract:\n{abstract_preview}")

            # Add source file information if available
            if "source_file" in result.payload:
                print(f"\nSource: {result.payload['source_file']}")

            print("-" * 80)

        if show_analytics:
            analytics = self.get_search_analytics()
            print("\nSearch Analytics:")
            print(f"Total searches this session: {analytics['total_searches']}")
            print(f"Average relevance score: {analytics['average_relevance']:.2f}")
            print(f"Unique papers shown: {analytics['unique_papers_shown']}")


async def main():
    searcher = PaperSearcher()

    print("Welcome to the Enhanced Paper Search System")
    print("Commands:")
    print("  'quit' or 'q': Exit the program")
    print("  'analytics': Show search analytics")
    print("  'clear': Clear shown papers history")

    while True:
        print("\nPaper Search Interface")
        print("Enter your query (or 'quit' to exit):")
        query = input("> ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            break
        elif query.lower() == "analytics":
            analytics = searcher.get_search_analytics()
            print("\nSearch Analytics:")
            print(json.dumps(analytics, indent=2))
            continue
        elif query.lower() == "clear":
            searcher.shown_papers.clear()
            print("Cleared shown papers history")
            continue
        elif not query:
            print("Please enter a valid query")
            continue

        print("\nHow many results would you like? (default: 3)")
        try:
            limit = int(input("> ") or "3")
        except ValueError:
            limit = 3

        try:
            results = await searcher.search_papers(query, limit)
            searcher.display_results(results, show_analytics=True)
        except Exception as e:
            print(f"Error during search: {e}")


if __name__ == "__main__":
    asyncio.run(main())
