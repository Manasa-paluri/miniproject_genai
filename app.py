import streamlit as st
import os, tempfile, uuid, time, json, base64, urllib.parse
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from supabase import create_client
import numpy as np
import requests
from bs4 import BeautifulSoup

# Load env variables and initialize clients
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Configure Gemini and page
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')
st.set_page_config(page_title="Document & Web Q&A with Gemini", layout="wide")

# Initialize session state variables
for key, default in {
    'chat_history': [], 'text_chunks': [], 'document_loaded': False,
    'document_name': None, 'document_type': None, 'embedder': None,
    'document_tag': None, 'pdf_content': None, 'all_chat_histories': {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if not GOOGLE_API_KEY:
    st.warning("Please set the GOOGLE_API_KEY in Streamlit secrets or environment variables!")

# Load sentence transformer model
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_data = pdf_file.read()
    st.session_state.pdf_content = pdf_data

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_data)
        temp_file_path = temp_file.name

    text_content = ""
    with open(temp_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n\n"

    os.unlink(temp_file_path)
    return text_content

# URL validation and cleaning
def is_valid_url(url):
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def clean_url(url):
    return url if url.startswith(('http://', 'https://')) else 'https://' + url

# Web scraping functions
def scrape_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        st.error(f"Error scraping URL {url}: {str(e)}")
        return None

def scrape_domain(domain, max_pages=10):
    domain = clean_url(domain)

    if not is_valid_url(domain):
        st.error("Invalid URL. Please enter a valid URL with format: example.com or https://example.com")
        return None

    visited, to_visit = set(), [domain]
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

                # Get and process page
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(current_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Clean and extract text
                    for script_or_style in soup(["script", "style"]):
                        script_or_style.decompose()

                    page_text = clean_text_for_storage(soup.get_text())
                    all_text += page_text + "\n\n--- Next Page ---\n\n"

                    # Find new links within same domain
                    domain_parts = urllib.parse.urlparse(domain).netloc
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        full_url = urllib.parse.urljoin(domain, href) if href.startswith('/') else href

                        if (is_valid_url(full_url) and domain_parts in full_url 
                            and full_url not in visited and full_url not in to_visit):
                            to_visit.append(full_url)

                visited.add(current_url)
                time.sleep(1)  # Be polite to servers

            except Exception as e:
                st.warning(f"Error processing {current_url}: {str(e)}")
                visited.add(current_url)

        progress_bar.progress(1.0)

    return all_text

# Text processing
def clean_text_for_storage(text):
    if text is None:
        return ""

    # Replace null bytes and filter control characters
    text = text.replace('\x00', '')
    cleaned_text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')

    # Ensure UTF-8 compatibility
    return cleaned_text.encode('utf-8', errors='ignore').decode('utf-8')

def split_text_into_chunks(text, chunk_size=1500, overlap=300):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:  # Only add substantial chunks
            chunks.append(chunk)
    return chunks

# Document management
def generate_document_tag():
    timestamp = int(time.time())
    random_id = str(uuid.uuid4())[:8]
    return f"doc_{timestamp}_{random_id}"

# Database operations
def save_chat_history_to_database(document_tag, chat_history):
    try:
        chat_history_json = json.dumps(chat_history)
        response = supabase.table("chat_histories").select("id").eq("document_tag", document_tag).execute()

        if response.data:
            # Update existing record
            record_id = response.data[0]["id"]
            supabase.table("chat_histories").update({
                "chat_history": chat_history_json,
                "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }).eq("id", record_id).execute()
        else:
            # Create new record
            supabase.table("chat_histories").insert({
                "document_tag": document_tag,
                "chat_history": chat_history_json,
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }).execute()

        return True
    except Exception as e:
        st.warning(f"Error saving chat history: {str(e)}")
        return False

def load_chat_history_from_database(document_tag):
    try:
        response = supabase.table("chat_histories").select("chat_history").eq("document_tag", document_tag).execute()

        if response.data and response.data[0]["chat_history"]:
            return json.loads(response.data[0]["chat_history"])
        return []
    except Exception as e:
        st.warning(f"Error loading chat history: {str(e)}")
        return []

def store_embeddings_in_supabase(text_chunks, embedder, source_type, source_name):
    with st.spinner(f"Generating embeddings for {len(text_chunks)} chunks..."):
        document_tag = generate_document_tag()
        st.session_state.document_tag = document_tag

        clean_chunks = [clean_text_for_storage(chunk) for chunk in text_chunks]
        embeddings = embedder.encode(clean_chunks)

        with st.spinner("Storing embeddings in database..."):
            batch_size = 50
            for i in range(0, len(clean_chunks), batch_size):
                batch_chunks = clean_chunks[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size].tolist()

                # Include document tag in chunk text
                batch_data = [
                    {
                        "chunk": f"[{source_type}: {source_name}] [tag: {document_tag}]\n{chunk}",
                        "embedding": embedding,
                    }
                    for chunk, embedding in zip(batch_chunks, batch_embeddings)
                ]

                try:
                    supabase.table("text_chunks").insert(batch_data).execute()
                except Exception as e:
                    st.error(f"Error storing embeddings: {str(e)}")
                    return False

    return True

def query_supabase(question, embedder, document_tag, top_k=6):
    # Generate embedding for the question
    question_embedding = embedder.encode([question])[0]

    # Retrieve chunks from Supabase
    response = supabase.table("text_chunks").select("*").execute()

    if not response.data:
        return "No documents found in the database."

    # Calculate similarities for chunks with matching document tag
    similarities = []
    for chunk in response.data:
        try:
            # Check if chunk belongs to our document
            chunk_text = chunk["chunk"]
            if f"[tag: {document_tag}]" not in chunk_text:
                continue

            # Parse embedding properly
            chunk_embedding = chunk["embedding"]
            if isinstance(chunk_embedding, str):
                try:
                    chunk_embedding = json.loads(chunk_embedding)
                except json.JSONDecodeError:
                    continue

            # Calculate cosine similarity
            chunk_embedding = np.array(chunk_embedding)
            question_embedding_np = np.array(question_embedding)

            norm_chunk = np.linalg.norm(chunk_embedding)
            norm_question = np.linalg.norm(question_embedding_np)

            if norm_chunk > 0 and norm_question > 0:
                similarity = np.dot(chunk_embedding, question_embedding_np) / (norm_chunk * norm_question)
                similarities.append((chunk["chunk"], similarity))

        except Exception:
            continue

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get top_k chunks
    top_chunks = [chunk for chunk, _ in similarities[:top_k]]

    if top_chunks:
        cleaned_chunks = [chunk.replace(f"[tag: {document_tag}]", "").strip() for chunk in top_chunks]
        context = "\n\n".join(cleaned_chunks)
        return context
    else:
        return "Could not find relevant information in the document."

def get_gemini_response(question, context):
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
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}. Please try a different question or check your API key."

def extract_document_tags_from_database():
    try:
        response = supabase.table("text_chunks").select("chunk").execute()
        if not response.data:
            return []

        import re
        tags = set()
        for item in response.data:
            chunk = item.get("chunk", "")
            tag_match = re.search(r'\[tag: (doc_\d+_[a-f0-9]+)\]', chunk)
            if tag_match:
                tags.add(tag_match.group(1))

        return list(tags)
    except Exception as e:
        st.warning(f"Error extracting document tags: {str(e)}")
        return []

def extract_source_info_from_tag(tag):
    try:
        response = supabase.table("text_chunks").select("chunk").execute()
        for item in response.data:
            chunk = item.get("chunk", "")
            if f"[tag: {tag}]" in chunk:
                import re
                source_match = re.search(r'\[(pdf|web): ([^\]]+)\]', chunk, re.IGNORECASE)
                if source_match:
                    return {
                        "source_type": source_match.group(1).lower(),
                        "source_name": source_match.group(2)
                    }
        return None
    except Exception as e:
        st.warning(f"Error extracting source info: {str(e)}")
        return None

def clear_current_document_data():
    for key in ['chat_history', 'text_chunks', 'document_loaded', 'document_name', 
                'document_type', 'document_tag', 'pdf_content']:
        st.session_state[key] = [] if key == 'chat_history' or key == 'text_chunks' else None
    st.session_state.document_loaded = False

# Database setup
def ensure_chat_histories_table_exists():
    try:
        supabase.table("chat_histories").select("id").limit(1).execute()
        return True
    except Exception as e:
        if "relation" in str(e) and "does not exist" in str(e):
            st.error("Please create a chat_histories table in your Supabase database with columns: id, document_tag, chat_history, created_at, updated_at")
            return False
        else:
            st.warning(f"Error checking chat_histories table: {str(e)}")
            return False

# Check database tables
ensure_chat_histories_table_exists()

# Load embedding model
if st.session_state.embedder is None:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedder = load_sentence_transformer()

with st.sidebar:
    st.header("Upload or Enter URL")

    input_method = st.radio("Choose input method:", ["PDF Upload", "Website URL"])

    if input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

        if uploaded_file is not None and (not st.session_state.document_loaded or 
                                         st.session_state.document_type != "pdf" or 
                                         st.session_state.document_name != uploaded_file.name):
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    text_content = extract_text_from_pdf(uploaded_file)
                    chunks = split_text_into_chunks(text_content)
                    st.info(f"Document processed: {len(chunks)} text chunks extracted")

                    clear_current_document_data()

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
                    scraped_text = scrape_domain(url_input, max_pages=max_pages)

                    if scraped_text:
                        chunks = split_text_into_chunks(scraped_text)
                        st.info(f"Website scraped: {len(chunks)} text chunks extracted")

                        clear_current_document_data()

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
            if st.session_state.chat_history:
                save_chat_history_to_database(st.session_state.document_tag, st.session_state.chat_history)
            clear_current_document_data()
            st.rerun()

    # Document history
    st.header("Document History")
    try:
        document_tags = extract_document_tags_from_database()

        if document_tags:
            st.write("Previously processed documents:")
            for tag in document_tags:
                source_info = extract_source_info_from_tag(tag)
                if source_info:
                    source_name = source_info["source_name"]
                    source_type = source_info["source_type"]

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{source_name} ({source_type})")
                    with col2:
                        if st.button("Load", key=f"load_{tag}"):
                            # Save current chat history before switching
                            if st.session_state.document_loaded and st.session_state.document_tag and st.session_state.chat_history:
                                save_chat_history_to_database(st.session_state.document_tag, st.session_state.chat_history)

                            # Load new document
                            st.session_state.document_tag = tag
                            st.session_state.document_name = source_name
                            st.session_state.document_type = source_type
                            st.session_state.document_loaded = True

                            # Load existing chat history
                            chat_history = load_chat_history_from_database(tag)
                            st.session_state.chat_history = chat_history if chat_history else []
                            st.session_state.pdf_content = None

                            st.rerun()
        else:
            st.write("No documents processed yet.")
    except Exception as e:
        st.warning(f"Could not retrieve document history: {str(e)}")

# ----- MAIN CHAT UI -----
if st.session_state.document_loaded:
    # Display document info and PDF viewer if applicable
    if st.session_state.document_type == "pdf":
        col1, col2 = st.columns([6, 2])
        with col1:
            st.subheader(f"Chatting with: {st.session_state.document_name}")
        with col2:
            if st.session_state.pdf_content:
                b64_pdf = base64.b64encode(st.session_state.pdf_content).decode()
                pdf_display = f'<a href="data:application/pdf;base64,{b64_pdf}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:8px 16px;border:none;border-radius:4px;cursor:pointer;">View PDF in New Tab</button></a>'
                st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.subheader(f"Chatting with: {st.session_state.document_name}")

    # Chat history display
    chat_container = st.container(height=400, border=True)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Chat history download
    if st.session_state.chat_history:
        chat_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_history])
        file_name = f"chat_history_{st.session_state.document_name.replace(' ', '_')}.txt"

        st.download_button(
            label="Download Chat History",
            data=chat_text.encode(),
            file_name=file_name,
            mime="text/plain",
        )

    # Chat input and response handling
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
                # Get context and response
                context = query_supabase(user_question, st.session_state.embedder, st.session_state.document_tag)
                response = get_gemini_response(user_question, context)

                # Display and save response
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                save_chat_history_to_database(st.session_state.document_tag, st.session_state.chat_history)
else:
    st.info("Please upload a PDF document or enter a website URL to start asking questions.")

# Footer
st.markdown("---")
st.caption("Document & Web Q&A System using Supabase and Google Gemini")
