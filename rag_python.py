import logging
from typing import List
from pprint import pprint
import chromadb
import openai
from chromadb.config import Settings
import os
import base64

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class ChromaRAG:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8001,
        collection_name: str = "stripe_docs"
    ):
        logger.debug(
            "Initializing ChromaRAG with host=%s, port=%d, collection_name=%s",
            host, port, collection_name
        )

        self.chroma_user = os.environ.get("BITNIMBUS_CHROMA_USER", "")
        self.chroma_password = os.environ.get("BITNIMBUS_CHROMA_PASSWORD", "")
        auth_token = base64.b64encode(f"{self.chroma_user}:{self.chroma_password}".encode()).decode()

        # If using a remote Chroma server
        self.chroma_client = chromadb.HttpClient(host=host, port=port,
                                                headers={
                                                     "Authorization": f"Basic {auth_token}"}
                                                )
        
        # If local, uncomment:
        # self.chroma_client = chromadb.Client(
        #     Settings(persist_directory="./my_chroma_db")
        # )

        logger.debug("Retrieving or creating the '%s' collection in Chroma...", collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Embeddings for Stripe documentation"}
        )
        logger.info("Successfully connected to Chroma and obtained the collection.")

    def generate_embedding(self, text: str):
        """Generate an embedding for a single text."""
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']

    def query_collection(self, query_texts: List[str], n_results: int = 5):
        """
        Query the existing collection with a list of string queries,
        using your own embeddings (OpenAI).
        """
        logger.debug("query_collection called with query_texts=%s, n_results=%d",
                     query_texts, n_results)
        try:
            # Generate an embedding for each query
            query_embeddings = [self.generate_embedding(q) for q in query_texts]
            
            # Pass those embeddings to Chroma
            result = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results
            )
            logger.debug(
                "Query completed successfully. Result count for first query: %d",
                len(result['ids'][0]) if result['ids'] else 0
            )
            return result
        except Exception as e:
            logger.error("Error querying collection: %s", e)
            raise

    def summarize(self, doc_texts: List[str]) -> str:
            """
            Summarize the retrieved RAG documents into a single, human-readable answer.
            Uses the OpenAI ChatCompletion API for a more conversational summary.
            """
            # Concatenate or otherwise combine documents
            combined_docs = "\n\n".join(doc_texts)

            logger.debug("Summarizing retrieved text. Total length: %d characters", len(combined_docs))

            # Example: Use gpt-3.5-turbo or a similar ChatCompletion model
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Summarize text for a user, "
                            "providing a concise, readable response."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following RAG documents:\n\n{combined_docs}"
                    }
                ],
                temperature=0.7
            )
            summary = response.choices[0].message["content"].strip()
            logger.debug("Obtained summary from OpenAI.")
            return summary


def main():
    logger.debug("Starting main function to demonstrate ChromaRAG usage.")
    chroma_rag = ChromaRAG(
        host="https://178-156-154-235.bitnimbususercontent.ai",
        port=443,
        collection_name="stripe_docs"
    )

    query_texts = ["How to set products in Stripe platform  ?"]
    logger.info("Sending query to ChromaRAG for texts: %s", query_texts)

    try:
        # Query Chroma
        results = chroma_rag.query_collection(query_texts, n_results=3)
        logger.info("Query results received.")
        pprint(results)

        # Extract documents (flatten nested lists)
        # The 'documents' field from Chroma typically is a list-of-lists structure
        # for each query, so we flatten it below:
        # all_docs = [doc for query_docs in results["documents"] for doc in query_docs]
        all_docs = []
        for i, query_docs in enumerate(results["documents"]):
            for j, doc in enumerate(query_docs):
                
                # Print some additional info about each doc
                snippet = doc[:100] + "..." if len(doc) > 100 else doc
                logger.debug("Document %d-%d [length=%d]: %s", i, j, len(doc), snippet)
                all_docs.append(doc)

        # Summarize the retrieved documents
        # summary = chroma_rag.summarize(all_docs)
        # logger.info("Summary:\n%s", summary)
        # print("\nFinal Summarized Answer:\n", summary)

    except Exception as e:
        logger.error("Failed to query Chroma or summarize results: %s", e)

if __name__ == "__main__":
    main()
