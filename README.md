# Pingbhai RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot for the Pingbhai electronics marketplace.

## Features
- **Domain Specific:** Strictly guarded to answer only Pingbhai-related queries.
- **RAG Architecture:** Uses FAISS vector store and OpenAI Embeddings.
- **Knowledge Base:** Structured JSON extracted from equipment and company pages.
- **Guardrails:** Pre-computation filtering and System Prompt constraints.

## Setup Instructions

1. **Clone the repository**

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt