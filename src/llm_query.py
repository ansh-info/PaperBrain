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
        self.model_name = "llama3.2:1b"
        self.search_history = []
        self.shown_papers = set()  # Track shown papers
        self.conversation_history = []  # Track Q&A history

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

Please provide a comprehensive answer in the following format:

MAIN ANSWER:
A clear, concise answer addressing the main question. Reference papers using [P1], [P2], etc.

KEY POINTS:
• Point 1 [P1]: Specific finding or contribution
• Point 2 [P2]: Specific finding or contribution
• Point 3 [P1, P3]: Specific finding or contribution
(Add more points as needed, always with clear paper references)

PAPER CITATIONS:
For each paper discussed above:
[P1] Title and key contributions discussed
[P2] Title and key contributions discussed
[P3] Title and key contributions discussed

Always use [P1], [P2], etc. consistently throughout the response.
"""

        try:
            async with httpx.AsyncClient() as client:
                print(f"Ensuring {self.model_name} model is available...")
                await client.post(
                    "http://localhost:11434/api/pull",
                    json={"name": self.model_name},
                    timeout=60.0,
                )

                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                    timeout=60.0,
                )

                if response.status_code != 200:
                    error_text = await response.text()
                    raise Exception(f"Error from Ollama API: {error_text}")

                response_data = response.json()
                if "error" in response_data:
                    raise Exception(response_data["error"])

                self.conversation_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "query": query,
                        "response": response_data.get("response", ""),
                        "num_papers": len(context.split("Title:")) - 1,
                    }
                )

                return response_data.get("response", "No response generated")

        except httpx.TimeoutException:
            return "Error: Request timed out. Please try again."
        except Exception as e:
            return f"Error: Unable to generate response. Details: {str(e)}"

    def format_section(self, text: str, section: str) -> str:
        """Format a section of the response with proper indentation and wrapping."""
        lines = []
        for line in text.split("\n"):
            wrapped = textwrap.fill(
                line, width=80, initial_indent="    ", subsequent_indent="    "
            )
            lines.append(wrapped)
        return f"{section}:\n" + "\n".join(lines)

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

    async def search_papers(self, query: str, limit: int = 3) -> List[Dict]:
        try:
            query_embedding = await self.get_embedding(query)

            # Search with higher limit to account for duplicates
            search_limit = limit * 2
            search_results = self.client.search(
                collection_name="papers",
                query_vector=query_embedding,
                limit=search_limit,
            )

            # Filter out previously shown papers
            unique_results = []
            for result in search_results:
                paper_id = result.payload["title"]
                if paper_id not in self.shown_papers:
                    self.shown_papers.add(paper_id)
                    unique_results.append(result)
                    if len(unique_results) >= limit:
                        break

            # Log the search
            self.log_search(query, unique_results)

            return unique_results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def log_search(self, query: str, results: List[Dict]):
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

    def get_analytics(self):
        """Get comprehensive analytics about system usage"""
        if not self.search_history:
            return "No searches performed yet."

        analytics = {
            "total_searches": len(self.search_history),
            "total_unique_papers": len(self.shown_papers),
            "total_questions": len(self.conversation_history),
            "avg_papers_per_query": (
                sum(qa["num_papers"] for qa in self.conversation_history)
                / len(self.conversation_history)
                if self.conversation_history
                else 0
            ),
            "avg_relevance_score": sum(h["avg_relevance"] for h in self.search_history)
            / len(self.search_history),
        }
        return analytics

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
        """
        Display the LLM response and search results in a well-formatted structure.

        Args:
            results: List of search results from Qdrant
            llm_response: String response from the LLM
        """
        if llm_response.startswith("Error:"):
            print("\nError in AI Response:")
            print("=" * 80)
            print(llm_response)
            return

        print("\nAI Response:")
        print("=" * 80)

        sections = {"MAIN ANSWER": [], "KEY POINTS": [], "PAPER CITATIONS": []}

        current_section = None
        current_content = []

        # Parse the response into sections
        for line in llm_response.split("\n"):
            line = line.strip()

            upper_line = line.upper()
            if any(section in upper_line for section in sections.keys()):
                if current_section and current_content:
                    sections[current_section] = current_content
                current_section = next(
                    (k for k in sections.keys() if k in upper_line), None
                )
                current_content = []
            elif current_section and line:
                # Remove any existing bullet points or asterisks
                line = line.lstrip("•").lstrip("*").strip()
                # Replace numbered citations with P-style citations
                for i in range(1, 10):
                    line = line.replace(f"[{i}]", f"[P{i}]")
                    line = line.replace(f"paper {i}", f"paper [P{i}]")
                    line = line.replace(f"Paper {i}", f"Paper [P{i}]")
                current_content.append(line)

        # Add the last section
        if current_section and current_content:
            sections[current_section] = current_content

        # Display Main Answer
        print("\nMAIN ANSWER:")
        print("-" * 40)
        main_answer_text = " ".join(sections.get("MAIN ANSWER", []))
        print(
            textwrap.fill(
                main_answer_text,
                width=80,
                initial_indent="    ",
                subsequent_indent="    ",
            )
        )

        # Display Key Points
        print("\nKEY POINTS:")
        print("-" * 40)
        for line in sections.get("KEY POINTS", []):
            if line.strip():
                # Add citation reference if missing
                if not any(f"[P{i}]" in line for i in range(1, len(results) + 1)):
                    line += " [P1]"  # Default to P1 if no citation is present

                wrapped_text = textwrap.fill(
                    line, width=80, initial_indent="    • ", subsequent_indent="      "
                )
                print(wrapped_text)

        # Display Paper Citations
        print("\nPAPER CITATIONS:")
        print("-" * 40)
        for i, result in enumerate(results, 1):
            title = result.payload["title"]
            score = result.score
            source = result.payload.get("source_file", "N/A")

            print(f"    [P{i}] {title}")
            print(f"         Relevance Score: {score:.2f}")
            print(f"         Source: {source}")

            if "abstract" in result.payload:
                abstract = result.payload["abstract"]
                if len(abstract) > 200:
                    abstract = abstract[:200] + "..."
                wrapped_abstract = textwrap.fill(
                    abstract,
                    width=76,
                    initial_indent="         Abstract: ",
                    subsequent_indent="                   ",
                )
                print(wrapped_abstract)
            print()

        # Display Relevant Papers section
        if results:
            print("\nRelevant Papers:")
            print("=" * 80)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Title: {result.payload['title']}")
                print(f"Relevance Score: {result.score:.2f}")
                print(
                    f"Relevance Explanation: {self.explain_relevance_score(result.score)}"
                )

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
                print(f"\nAbstract Preview:\n{abstract_preview}")

                if "source_file" in result.payload:
                    print(f"\nSource: {result.payload['source_file']}")
                print("-" * 80)
        else:
            print("\nNo relevant papers found.")


async def main():
    searcher = PaperSearcher()

    print("Welcome to the Enhanced Research Paper Q&A System")
    print("Commands:")
    print("  'quit' or 'q': Exit the program")
    print("  'analytics': Show system analytics")
    print("  'clear': Clear paper history")
    print("  'history': Show question history")

    while True:
        print("\nResearch Paper Q&A System")
        print("Enter your question (or command):")
        query = input("> ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            break
        elif query.lower() == "analytics":
            analytics = searcher.get_analytics()
            print("\nSystem Analytics:")
            print(json.dumps(analytics, indent=2))
            continue
        elif query.lower() == "clear":
            searcher.shown_papers.clear()
            print("Cleared paper history")
            continue
        elif query.lower() == "history":
            print("\nQuestion History:")
            for i, qa in enumerate(searcher.conversation_history[-5:], 1):
                print(f"\n{i}. Query: {qa['query']}")
                print(f"   Time: {qa['timestamp']}")
                print(f"   Papers Used: {qa['num_papers']}")
            continue
        elif not query:
            print("Please enter a valid query")
            continue

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
