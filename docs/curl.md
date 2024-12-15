# How to chekc if olllma is running and your model is loaded?

```bash
# Make a curl request to this endpoint - to query llama3
curl -X POST http://localhost:11434/api/generate -d '{"model": "llama3.2:1b", "prompt": "Hello!", "stream": false}'
```

```bash
# Make a curl request to this endpoint - to return vectors of the prompt specified to the embedding model
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "The sky is blue because of Rayleigh scattering"
}'
```
