# Stripe Documentation Scraper and RAG System

This project scrapes Stripe documentation, generates text embeddings using OpenAI, and stores them in a Chroma vector database. It also includes a Retrieval-Augmented Generation (RAG) system to query the stored embeddings and summarize the results using OpenAI's GPT-3.5-turbo model.

## Features

- Scrapes Stripe documentation using Playwright for dynamic content rendering.
- Generates text embeddings using OpenAI's `text-embedding-ada-002` model.
- Stores embeddings in a Chroma vector database.
- Queries the Chroma database and summarizes the results using OpenAI's GPT-3.5-turbo model.

## Installation

### Prerequisites

- Python 3.7 or higher
- Node.js (for Playwright)
- OpenAI API key
- Chroma server (local or remote)

### Step-by-Step Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/rag_example_python.git
   cd rag_example_python

2. **Set up a virtual environment:**
```python
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. **Install Python dependencies:**

```python
pip install -r requirements.txt
```
4. **Install Playwright and its dependencies:**

```python
pip install playwright
playwright install
```

5. **Set Environment Variables:**

```
export OPENAI_API_KEY="your_openai_api_key"
export BITNIMBUS_CHROMA_USER="your_chroma_user"
export BITNIMBUS_CHROMA_PASSWORD="your_chroma_password"
```

### Scraping and Storing Stripe Documentation


```
python stripe_scraper.py
```

### Querying and Summarizing Results

```
python rag_python.py
```