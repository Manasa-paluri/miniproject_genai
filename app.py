import streamlit as st
import os
import tempfile
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from supabase import create_client
import numpy as np
import json
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import uuid

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Set page configuration
st.set_page_config(page_title="Document & Web Q&A with Gemini", layout="wide")

# Initialize session state variables if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
if 'document_name' not in st.session_state:
    st.session_state.document_name = None
if 'document_type' not in st.session_state:
    st.session_state.document_type = None  # "pdf" or "web"
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'document_tag' not in st.session_state:
    st.session_state.document_tag = None

if not GOOGLE_API_KEY:
    st.warning("Please set the GOOGLE_API_KEY in Streamlit secrets or environment variables!")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Initialize sentence transformer model
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name

    text_content = ""
    with open(temp_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text() + "\n\n"

    os.unlink(temp_file_path)
    return text_content

def is_valid_url(url):
    """Check if the provided string is a valid URL"""
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def clean_url(url):
    """Ensure URL has proper scheme"""
    if not url.startswith(('http://', 'https://')):
        return 'https://' + url
    return url

def scrape_url(url):
    """Scrape text content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        st.error(f"Error scraping URL {url}: {str(e)}")
        return None

def clean_text_for_storage(text):
    """Clean text of problematic Unicode characters and null bytes"""
    if text is None:
        return ""
        
    # Replace null bytes
    text = text.replace('\x00', '')
    
    # Handle other problematic characters
    cleaned_text = ""
    for char in text:
        if ord(char) < 32 and char not in '\n\r\t':
            # Skip control characters except newlines, returns, and tabs
            continue
        cleaned_text += char
    
    # Additional cleaning for PostgreSQL compatibility
    cleaned_text = cleaned_text.encode('utf-8', errors='ignore').decode('utf-8')
    
    return cleaned_text

def split_text_into_chunks(text, chunk_size=1500, overlap=300):
    """Split text into overlapping chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:  # Only add chunks that have substantial content
            chunks.append(chunk)
    return chunks

def scrape_domain(domain, max_pages=10):
    """Scrape multiple pages from a domain"""
    domain = clean_url(domain)
    base_url = domain

    if not is_valid_url(base_url):
        st.error("Invalid URL. Please enter a valid URL with format: example.com or https://example.com")
        return None

    visited = set()
    to_visit = [base_url]
    all_text = ""

    with st.spinner(f"Scraping website (0/{max_pages} pages)..."):
        progress_bar = st.progress(0)
        page_count = 0

        while to_visit and page_count < max_pages:
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue

            try:
                # Update progress
                page_count += 1
                progress_bar.progress(page_count / max_pages)
                st.spinner(f"Scraping website ({page_count}/{max_pages} pages)...")

                # Get page content
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(current_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Extract text
                    for script_or_style in soup(["script", "style"]):
                        script_or_style.decompose()
                    
                    # Get cleaned text
                    page_text = soup.get_text()
                    
                    # Clean the text of problematic characters
                    page_text = clean_text_for_storage(page_text)
                    
                    all_text += page_text + "\n\n--- Next Page ---\n\n"

                    # Find new links within same domain
                    domain_parts = urllib.parse.urlparse(base_url).netloc
                    for link in soup.find_all('a', href=True):
                        href = link['href']

                        # Handle relative URLs
                        if href.startswith('/'):
                            full_url = urllib.parse.urljoin(base_url, href)
                        else:
                            full_url = href

                        # Only add URLs from same domain
                        if is_valid_url(full_url) and domain_parts in full_url and full_url not in visited and full_url not in to_visit:
                            to_visit.append(full_url)

                visited.add(current_url)
                # Small delay to be polite to the server
                time.sleep(1)

            except Exception as e:
                st.warning(f"Error processing {current_url}: {str(e)}")
                visited.add(current_url)  # Mark as visited to avoid retrying

        progress_bar.progress(1.0)

    return all_text

def generate_document_tag():
    """Generate a unique document tag with timestamp and random ID"""
    timestamp = int(time.time())
    random_id = str(uuid.uuid4())[:8]
    return f"doc_{timestamp}_{random_id}"

def store_embeddings_in_supabase(text_chunks, embedder, source_type, source_name):
    """Store text chunks and their embeddings in Supabase with document tag in the chunk text"""
    with st.spinner(f"Generating embeddings for {len(text_chunks)} chunks..."):
        # Generate a document tag for this document
        document_tag = generate_document_tag()
        st.session_state.document_tag = document_tag
        
        # Clean all chunks before embedding
        clean_chunks = [clean_text_for_storage(chunk) for chunk in text_chunks]
        embeddings = embedder.encode(clean_chunks)

        with st.spinner("Storing embeddings in database..."):
            # Insert chunks in batches for efficiency
            batch_size = 50
            for i in range(0, len(clean_chunks), batch_size):
                batch_chunks = clean_chunks[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size].tolist()

                # Include document tag in the chunk text itself
                batch_data = [
                    {
                        "chunk": f"[{source_type}: {source_name}] [tag: {document_tag}]\n{chunk}",
                        "embedding": embedding
                    }
                    for chunk, embedding in zip(batch_chunks, batch_embeddings)
                ]

                try:
                    supabase.table("text_chunks").insert(batch_data).execute()
                except Exception as e:
                    st.error(f"Error storing embeddings in Supabase: {str(e)}")
                    return False

    return True

def query_supabase(question, embedder, document_tag, top_k=6):
    """Query Supabase to find relevant document chunks using tag filtering and cosine similarity"""
    # Generate embedding for the question
    question_embedding = embedder.encode([question])[0]  # Get the embedding vector

    # Retrieve all chunks from Supabase
    response = supabase.table("text_chunks").select("*").execute()

    # Check if data exists
    if not response.data:
        return "No documents found in the database."

    # Calculate similarities but only for chunks with matching document tag
    similarities = []
    for chunk in response.data:
        try:
            # Check if this chunk belongs to our document by tag
            chunk_text = chunk["chunk"]
            if f"[tag: {document_tag}]" not in chunk_text:
                continue
                
            # Parse the embedding properly - handle both list and string formats
            chunk_embedding = chunk["embedding"]
            if isinstance(chunk_embedding, str):
                try:
                    chunk_embedding = json.loads(chunk_embedding)
                except json.JSONDecodeError:
                    continue

            # Convert to numpy arrays for calculation
            chunk_embedding = np.array(chunk_embedding)
            question_embedding_np = np.array(question_embedding)

            # Calculate cosine similarity (dot product of normalized vectors)
            norm_chunk = np.linalg.norm(chunk_embedding)
            norm_question = np.linalg.norm(question_embedding_np)

            if norm_chunk > 0 and norm_question > 0:  # Avoid division by zero
                similarity = np.dot(chunk_embedding, question_embedding_np) / (norm_chunk * norm_question)
                similarities.append((chunk["chunk"], similarity))

        except Exception as e:
            continue

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get top_k chunks
    top_chunks = [chunk for chunk, _ in similarities[:top_k]]

    # Return combined context from top chunks
    if top_chunks:
        
        cleaned_chunks = [chunk.replace(f"[tag: {document_tag}]", "").strip() for chunk in top_chunks]
        context = "\n\n".join(cleaned_chunks)
        return context
    else:
        return "Could not find relevant information in the document."

def get_gemini_response(question, context):
    """Get response from Gemini model based on context"""
    prompt = f"""
    You are a helpful assistant answering questions based on the provided document or web content.

    Context:
    {context}

    User question: {question}

    Instructions:
    1. Answer based ONLY on the information in the context above
    2. If the context doesn't contain information to answer the question, respond with:
       "The document doesn't contain information to answer this question."
    3. Be concise but complete in your answer
    4. You may cite specific parts of the document to support your answer
    5. If the context contains source information in [PDF: filename] or [web: URL] format, you can 
       mention which source the information came from

    Example:

Context:
- " jman is commercially-focused data partner for PE funds and their portfolio companies. "

User Question: " on Which area jman is the  working?"

Expected Answer:
"jman is working in the area of commercial data partnership for PE funds and their portfolio companies."

User Question: "What is the position of jman in it industry of india ?"

Expected Answer:
"The document doesn't contain information to answer this question."
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}. Please try a different question or check your API key."

def extract_document_tags_from_database():
    """Extract all document tags from the database chunks"""
    try:
        response = supabase.table("text_chunks").select("chunk").execute()
        if not response.data:
            return []
        
        tags = set()
        for item in response.data:
            chunk = item.get("chunk", "")
            # Find tag pattern [tag: doc_timestamp_uuid]
            import re
            tag_match = re.search(r'\[tag: (doc_\d+_[a-f0-9]+)\]', chunk)
            if tag_match:
                tags.add(tag_match.group(1))
        
        return list(tags)
    except Exception as e:
        st.warning(f"Error extracting document tags: {str(e)}")
        return []

def extract_source_info_from_tag(tag):
    """Find source information for a given document tag"""
    try:
        response = supabase.table("text_chunks").select("chunk").execute()
        for item in response.data:
            chunk = item.get("chunk", "")
            if f"[tag: {tag}]" in chunk:
                # Extract source information [pdf: filename] or [web: url]
                import re
                source_match = re.search(r'\[(pdf|web): ([^\]]+)\]', chunk, re.IGNORECASE)
                if source_match:
                    source_type = source_match.group(1).lower()
                    source_name = source_match.group(2)
                    return {
                        "source_type": source_type,
                        "source_name": source_name
                    }
        return None
    except Exception as e:
        st.warning(f"Error extracting source info: {str(e)}")
        return None

def clear_current_document_data():
    """Clear session state related to the current document without affecting database"""
    st.session_state.chat_history = []
    st.session_state.text_chunks = []
    st.session_state.document_loaded = False
    st.session_state.document_name = None
    st.session_state.document_type = None
    st.session_state.document_tag = None

# UI Components
st.title("Document & Web Question Answering System")

# Load the embedder
if st.session_state.embedder is None:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedder = load_sentence_transformer()

# Sidebar for input method selection
with st.sidebar:
    st.header("Upload or Enter URL")

    # Radio button to choose input method
    input_method = st.radio("Choose input method:", ["PDF Upload", "Website URL"])

    if input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

        if uploaded_file is not None and (not st.session_state.document_loaded or 
                                         st.session_state.document_type != "pdf" or 
                                         st.session_state.document_name != uploaded_file.name):
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    # Extract text from PDF
                    text_content = extract_text_from_pdf(uploaded_file)

                    # Split text into chunks
                    chunks = split_text_into_chunks(text_content)
                    st.info(f"Document processed: {len(chunks)} text chunks extracted")

                    # Clear current document context (without deleting data)
                    clear_current_document_data()
                    
                    # Store embeddings in Supabase with document tag
                    if store_embeddings_in_supabase(chunks, st.session_state.embedder, "pdf", uploaded_file.name):
                        st.session_state.text_chunks = chunks
                        st.session_state.document_loaded = True
                        st.session_state.document_type = "pdf"
                        st.session_state.document_name = uploaded_file.name
                        st.success(f"Successfully uploaded and indexed: {uploaded_file.name}")

    else:  # Website URL
        url_input = st.text_input("Enter website URL or domain:", placeholder="example.com or https://example.com")
        max_pages = st.slider("Maximum pages to scrape:", min_value=1, max_value=20, value=5)

        if url_input and st.button("Scrape Website"):
            if not is_valid_url(clean_url(url_input)):
                st.error("Invalid URL. Please enter a valid domain or URL.")
            else:
                with st.spinner("Scraping website..."):
                    # Scrape the website
                    scraped_text = scrape_domain(url_input, max_pages=max_pages)

                    if scraped_text:
                        # Split text into chunks
                        chunks = split_text_into_chunks(scraped_text)
                        st.info(f"Website scraped: {len(chunks)} text chunks extracted")

                        # Clear current document context (without deleting data)
                        clear_current_document_data()
                        
                        # Store embeddings in Supabase with document tag
                        if store_embeddings_in_supabase(chunks, st.session_state.embedder, "web", url_input):
                            st.session_state.text_chunks = chunks
                            st.session_state.document_loaded = True
                            st.session_state.document_type = "web"
                            st.session_state.document_name = url_input
                            st.success(f"Successfully scraped and indexed: {url_input}")

    # Clear button
    if st.session_state.document_loaded:
        st.success(f"Current source: {st.session_state.document_name} ({st.session_state.document_type})")
        if st.button("Clear Current Source"):
            clear_current_document_data()
            st.rerun()
            
    # Show document history
    st.header("Document History")
    try:
        # Get document tags
        document_tags = extract_document_tags_from_database()
        
        if document_tags:
            st.write("Previously processed documents:")
            for tag in document_tags:
                # Get source info
                source_info = extract_source_info_from_tag(tag)
                if source_info:
                    source_name = source_info["source_name"]
                    source_type = source_info["source_type"]
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{source_name} ({source_type})")
                    with col2:
                        if st.button("Load", key=f"load_{tag}"):
                            st.session_state.document_tag = tag
                            st.session_state.document_name = source_name
                            st.session_state.document_type = source_type
                            st.session_state.document_loaded = True
                            st.session_state.chat_history = []
                            st.rerun()
        else:
            st.write("No documents processed yet.")
    except Exception as e:
        st.warning(f"Could not retrieve document history: {str(e)}")

# Main chat interface
if st.session_state.document_loaded:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input(f"Ask a question about {st.session_state.document_type}: {st.session_state.document_name}")
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Display user message
        with st.chat_message("user"):
            st.write(user_question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating response..."):
                # Retrieve relevant context using Supabase - specific to current document
                context = query_supabase(user_question, st.session_state.embedder, st.session_state.document_tag)

                # Get response from Gemini
                response = get_gemini_response(user_question, context)

                # Display response
                st.write(response)

                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
else:
    st.info("Please upload a PDF document or enter a website URL to start asking questions.")

# Footer
st.markdown("---")
st.caption("Document & Web Q&A System using Supabase and Google Gemini")
