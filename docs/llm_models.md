```bash
# Start Ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Pull models
docker exec -it ollama ollama pull mistral
docker exec -it ollama ollama pull nomic-embed-text
```

Then start Qdrant:

```bash
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage --name qdrant qdrant/qdrant
```
