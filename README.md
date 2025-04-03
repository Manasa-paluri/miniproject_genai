

# Document & Web Q&A System Flow

## System Overview
This application allows users to upload PDF documents or provide website URLs, then ask questions about their content using Google's Gemini AI.

## Core Workflow

1. **Content Input**
   - Upload PDF documents
   - Enter website URLs for scraping (with configurable page limit)

2. **Document Processing**
   - PDFs: Extract text content using PyPDF2
   - Websites: Scrape content with BeautifulSoup
   - Split content into manageable chunks

3. **Embedding & Storage**
   - Generate vector embeddings using SentenceTransformer
   - Store text chunks and embeddings in Supabase database
   - Assign unique document tags for identification

4. **Question-Answering**
   - User submits question through chat interface
   - System finds relevant document chunks via embedding similarity search
   - Google Gemini AI generates contextual answers based on retrieved chunks

5. **Session Management**
   - Save and load chat histories from database
   - Switch between previously processed documents
   - Download conversation history

## Technical Components
- **Frontend**: Streamlit web interface
- **AI Models**: Google Gemini for response generation, SentenceTransformer for embeddings
- **Database**: Supabase for vector storage and chat history
- **Document Processing**: PyPDF2, BeautifulSoup
