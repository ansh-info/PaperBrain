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
    # More explicit and structured prompt
    prompt = f"""You are a research assistant helping to analyze scientific papers. Please provide a comprehensive answer to the following question based on the provided research papers.

Question: {query}

Available papers:
{context}

Please structure your response in the following format:

**Main Answer**
[Provide a concise but thorough answer to the question, synthesizing information from the papers]

**Key Points**
* [First key finding or point from the papers]
* [Second key finding or point from the papers]
* [Additional key points as needed]

**Paper Citations**
* [Paper 1]: [Brief description of its contribution]
* [Paper 2]: [Brief description of its contribution]
* [Additional papers as needed]

**Limitations**
* [First limitation or gap in the available information]
* [Second limitation or gap in the available information]
* [Additional limitations as needed]

References:
[List the full citations of all papers used]

Remember to:
1. Be specific and cite papers when making claims
2. Focus on the main findings relevant to the question
3. Acknowledge any limitations in the available information"""

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
                    "top_k": 40,  # Added for better response quality
                    "repeat_penalty": 1.1,  # Helps avoid repetition
                    "max_tokens": 2048,  # Ensure enough tokens for full response
                },
                timeout=120.0,  # Increased timeout for longer responses
            )

            if response.status_code != 200:
                error_text = await response.text()
                raise Exception(f"Error from Ollama API: {error_text}")

            response_data = response.json()
            if "error" in response_data:
                raise Exception(response_data["error"])

            # Log the Q&A interaction
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response_data.get("response", ""),
                "num_papers": len(context.split("Title:")) - 1,
            })

            # Post-process the response to ensure proper formatting
            response_text = response_data.get("response", "")
            if "**Main Answer**" not in response_text:
                # If response isn't properly formatted, restructure it
                response_text = f"""**Main Answer**
{response_text}

**Key Points**
* Key findings from the response

**Paper Citations**
* Citations from the provided papers

**Limitations**
* Potential limitations of the analysis"""

            return response_text

    except httpx.TimeoutException:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        return f"Error: Unable to generate response. Details: {str(e)}" 

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
        if llm_response.startswith("Error:"):
            print("\nError in AI Response:")
            print("=" * 80)
            print(llm_response)
        else:
            print("\nAI Response:")
            print("=" * 80)

            # Split response into sections based on markdown-style headers
            sections = llm_response.split("**")

            for i, section in enumerate(sections):
                if i == 0:  # Skip first empty section
                    continue

                # Process each section
                if section.startswith("Main Answer"):
                    print("\nMAIN ANSWER")
                    print("-" * 40)
                    content = section.replace("Main Answer**", "").strip()
                    print(textwrap.fill(content, width=80))

                elif section.startswith("Key Points"):
                    print("\nKEY POINTS")
                    print("-" * 40)
                    content = section.replace("Key Points**", "").strip()
                    # Split and format bullet points
                    points = content.split("*")
                    for point in points:
                        if point.strip():
                            print(
                                textwrap.fill(
                                    point.strip(),
                                    width=80,
                                    initial_indent="• ",
                                    subsequent_indent="  ",
                                )
                            )

                elif section.startswith("Paper Citations"):
                    print("\nPAPER CITATIONS")
                    print("-" * 40)
                    content = section.replace("Paper Citations**", "").strip()
                    citations = content.split("*")
                    for citation in citations:
                        if citation.strip():
                            print(f"• {citation.strip()}")

                elif section.startswith("Limitations"):
                    print("\nLIMITATIONS")
                    print("-" * 40)
                    content = section.replace("Limitations**", "").strip()
                    limitations = content.split("*")
                    for limitation in limitations:
                        if limitation.strip():
                            print(
                                textwrap.fill(
                                    limitation.strip(),
                                    width=80,
                                    initial_indent="• ",
                                    subsequent_indent="  ",
                                )
                            )

            # Display references if present
            if "References:" in llm_response:
                print("\nREFERENCES")
                print("-" * 40)
                refs_section = llm_response.split("References:")[1].strip()
                refs = refs_section.split("[")
                for ref in refs:
                    if ref.strip():
                        print(
                            textwrap.fill(
                                f"[{ref.strip()}",
                                width=80,
                                initial_indent="",
                                subsequent_indent="  ",
                            )
                        )

        # Display Paper Results
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
