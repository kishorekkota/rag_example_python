import os
import openai
import chromadb
from chromadb.config import Settings
from bs4 import BeautifulSoup
from typing import List, Set, Tuple
from collections import deque
import base64

# We'll use Playwright for dynamic rendering
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

class StripeDocScraper:
    def __init__(
        self, 
        base_url: str = "https://docs.stripe.com/", 
        openai_api_key: str = None,
        chroma_host: str = "localhost",    
        chroma_port: int = 8001
    ):
        print("[DEBUG] Initializing StripeDocScraper...")
        self.base_url = base_url.rstrip('/')
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.chroma_user = os.environ.get("BITNIMBUS_CHROMA_USER", "")
        self.chroma_password = os.environ.get("BITNIMBUS_CHROMA_PASSWORD", "")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is not provided.")

        openai.api_key = self.openai_api_key

        auth_token = base64.b64encode(f"{self.chroma_user}:{self.chroma_password}".encode()).decode()


        print(f"[DEBUG] Connecting to Chroma at {chroma_host}:{chroma_port}")
        self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port,    
                                                headers={
                                                     "Authorization": f"Basic {auth_token}"}
                                                )

        print("[DEBUG] Accessing or creating 'stripe_docs' collection...")
        self.collection = self.chroma_client.get_or_create_collection(
            name="stripe_docs",
            metadata={"description": "Embeddings for Stripe documentation"}
        )
        print("[DEBUG] StripeDocScraper initialization complete.")

    def fetch_urls_from_page(self, page_url: str) -> List[str]:
        print(f"[DEBUG] Entering fetch_urls_from_page with page_url={page_url}")

        with sync_playwright() as p:
            print("[DEBUG] Launching Playwright browser...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            context.set_default_timeout(60000)  # 60 seconds default timeout
            page = context.new_page()

            print(f"[DEBUG] Navigating to {page_url}")
            try:
                page.goto(page_url, timeout=60000)
                # Wait for the DOM to be loaded, avoiding networkidle
                print("[DEBUG] Waiting for domcontentloaded...")
                page.wait_for_load_state("domcontentloaded", timeout=60000)

                print("[DEBUG] Retrieving content...")
                content = page.content()
                # print("[DEBUG] content: %s" % content)
            except PlaywrightTimeout as e:
                print(f"[DEBUG] Timed out fetching URLs from {page_url}: {e}")
                content = ""
            finally:
                browser.close()

        soup = BeautifulSoup(content, "html.parser")
        links = soup.find_all("a", href=True)
        print(f"[DEBUG] Found {len(links)} total links on this page.")
        doc_urls = []
        for link in links:
            
            href = link.get("href")
            print(f"[DEBUG] Found link: {href}")
            if href and href.startswith("/") and not href.startswith("#"):
                full_url = f"https://docs.stripe.com{href}" 
                doc_urls.append(full_url)
                print("[DEBUG] full_url....: %s" % full_url)

        unique_urls = list(set(doc_urls))
        print(f"[DEBUG] Found {len(unique_urls)} unique doc URLs on this page.")
        print("[DEBUG] Exiting fetch_urls_from_page")
        return unique_urls

    def scrape_page_text(self, url: str) -> str:
        print(f"[DEBUG] Entering scrape_page_text with url={url}")
        with sync_playwright() as p:
            print("[DEBUG] Launching Playwright browser for scraping...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            context.set_default_timeout(60000)

            page = context.new_page()
            print(f"[DEBUG] Navigating to {url}")
            try:
                page.goto(url, timeout=60000)
                # Wait until the DOM is loaded
                print("[DEBUG] Waiting for domcontentloaded...")
                page.wait_for_load_state("domcontentloaded", timeout=60000)

                print("[DEBUG] Extracting page content...")
                content = page.content()
            except PlaywrightTimeout as e:
                print(f"[DEBUG] Timed out scraping page {url}: {e}")
                content = ""
            finally:
                browser.close()

        soup = BeautifulSoup(content, "html.parser")
        main_content = soup.find("main")
        content_text = main_content.get_text(separator="\n") if main_content else ""
        cleaned_text = " ".join(content_text.split())
        print(f"[DEBUG] Extracted {len(cleaned_text)} characters from page.")
        print("[DEBUG] Exiting scrape_page_text")
        return cleaned_text

    def embed_text(self, text: str) -> List[float]:
        print("[DEBUG] Entering embed_text...")
        print(f"[DEBUG] Text length to embed: {len(text)} characters")
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = response["data"][0]["embedding"]
        print(f"[DEBUG] Generated embedding with {len(embedding)} dimensions.")
        print("[DEBUG] Exiting embed_text")
        return embedding

    def add_to_vector_db(self, doc_id: str, text: str,url: str):
        print(f"[DEBUG] Entering add_to_vector_db with doc_id={doc_id}")
        embedding = self.embed_text(text)
        print("[DEBUG] Adding document and embedding to collection...")
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{"url": url}]        )
        print(f"[DEBUG] Document added to vector DB. doc_id={doc_id}")
        print("[DEBUG] Exiting add_to_vector_db")

    def full_crawl_and_store(self, start_path: str = "", max_depth: int = 2):
        print(f"[DEBUG] Entering full_crawl_and_store with start_path='{start_path}', max_depth={max_depth}")
        start_url = self.base_url if not start_path else f"{self.base_url}/{start_path.lstrip('/')}"
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque()

        print(f"[DEBUG] Initializing crawl queue with start_url={start_url}")
        queue.append((start_url, 0))
        visited.add(start_url)

        while queue:
            current_url, depth = queue.popleft()
            print(f"[DEBUG] Crawling {current_url} at depth {depth}")

            try:
                doc_text = self.scrape_page_text(current_url)
                if doc_text:
                    print(f"[DEBUG] Splitting text into chunks for {current_url}")
                    chunks = chunk_text(doc_text, chunk_size=1000, chunk_overlap=200)
                    print(f"[DEBUG] Got {len(chunks)} chunks.")
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{current_url}--chunk{i}"
                        self.add_to_vector_db(doc_id=chunk_id, text=chunk,url=current_url)
                else:
                    print(f"[DEBUG] No text found for {current_url}.")
            except Exception as e:
                print(f"[DEBUG] Error scraping {current_url}: {e}")

            if depth < max_depth:
                try:
                    print(f"[DEBUG] Fetching sub-urls for {current_url}")
                    sub_urls = self.fetch_urls_from_page(current_url)
                    print(f"[DEBUG] {len(sub_urls)} sub-urls found for {current_url}")
                    for link in sub_urls:
                        if link not in visited:
                            visited.add(link)
                            print(f"[DEBUG] Enqueuing link={link} for depth {depth+1}")
                            queue.append((link, depth + 1))
                except Exception as e:
                    print(f"[DEBUG] Error fetching sub-urls from {current_url}: {e}")

        print("[DEBUG] Exiting full_crawl_and_store")


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    print("[DEBUG] Entering chunk_text")
    print(f"[DEBUG] Text size: {len(text)} characters. chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += (chunk_size - chunk_overlap)
    print(f"[DEBUG] Exiting chunk_text with {len(chunks)} chunks")
    return chunks


def main():
    print("[DEBUG] Entering main function")
    scraper = StripeDocScraper(
        base_url="https://docs.stripe.com/",
        openai_api_key=None,
        chroma_host="https://178-156-154-235.bitnimbususercontent.ai",
        chroma_port=443
    )

    print("[DEBUG] Starting full crawl...")
    scraper.full_crawl_and_store(start_path="", max_depth=2)
    print("[DEBUG] Exiting main function")

if __name__ == "__main__":
    main()
