How the relevance score is calculated in our pipeline, step by step:

1. **Vector Creation Stage**

- When we first index papers into Qdrant:
  ```python
  text = f"{title}\n{abstract}"  # We combine title and abstract
  embedding = await get_embedding(text)  # Get vector using nomic-embed-text model
  ```
- Each paper's title and abstract are combined and converted into a 768-dimensional vector
- These vectors capture semantic meaning through the nomic-embed-text model
- Each dimension represents some latent feature learned by the model

2. **Query Processing Stage**

- When a user makes a query:
  ```python
  query_embedding = await get_embedding(query)
  ```
- The query is converted into a vector of the same dimensionality (768)
- Uses the same nomic-embed-text model for consistency

3. **Similarity Calculation Stage**

- Qdrant calculates cosine similarity between the query vector and all paper vectors:
  ```python
  search_results = self.client.search(
      collection_name="papers",
      query_vector=query_embedding,
      limit=limit
  )
  ```
- Cosine similarity is calculated as:
  ```
  similarity = cos(θ) = (A·B)/(||A||·||B||)
  ```
  Where:
  - A is the query vector
  - B is each paper's vector
  - · represents dot product
  - ||A|| is the magnitude of vector A

4. **Score Interpretation**

- The resulting score ranges from -1 to 1, but in practice usually 0 to 1 because:
  - 1.0 = vectors point in same direction (perfect match)
  - 0.0 = vectors are perpendicular (unrelated)
  - -1.0 = vectors point in opposite directions (rarely happens)

For example:

```python
# If we have:
query = "Neural networks for time series prediction"
paper_title = "Deep learning approaches to time series forecasting"
paper_abstract = "This paper explores neural network architectures..."

# The process is:
1. query_vector = model.embed(query)  # e.g. [0.1, 0.3, -0.2, ...]
2. paper_vector = model.embed(f"{paper_title}\n{paper_abstract}")
3. similarity = cosine_similarity(query_vector, paper_vector)
4. score = 0.85  # High score because concepts are semantically similar
```

This is why you might see:

- Papers about similar topics scoring 0.7-0.9
- Somewhat related papers scoring 0.5-0.7
- Unrelated papers scoring below 0.5

The key advantages of this approach are:

1. Semantic understanding (not just keyword matching)
2. Language model's understanding of scientific concepts
3. Unified representation of queries and papers
4. Fast similarity search through Qdrant's vector index
