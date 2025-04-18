import streamlit as st
from rag_app import WebRAG
import time
import requests
import json
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="Web RAG Assistant",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-container {
        border-radius: 10px;
        padding: 20px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .user-message {
        background-color: #2e7bf3;
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #white;
        padding: 15px;
        border-radius: 15px;
        margin: 5px 0;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = WebRAG()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_processed' not in st.session_state:
    st.session_state.query_processed = False
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'scraped_urls' not in st.session_state:
    st.session_state.scraped_urls = []
if 'raw_results' not in st.session_state:
    st.session_state.raw_results = ""

# Function to reset chat history
def reset_chat_history():
    st.session_state.chat_history = []
    st.session_state.current_query = ""
    st.session_state.query_processed = False # Reset processing status
    st.session_state.scraped_urls = []
    st.session_state.raw_results = ""

# --- Web Search and Content Extraction Functions ---
def search_web(query, api_key):
    """Performs a web search using the Serper API."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        search_results = response.json()
        # Extract URLs from organic results
        urls = [result['link'] for result in search_results.get('organic', [])[:10]] # Get top 5 results
        return urls
    except requests.exceptions.RequestException as e:
        st.error(f"Error during web search: {e}")
        return []
    except json.JSONDecodeError:
        st.error("Error decoding search results.")
        return []

def fetch_and_extract_text(urls):
    """Fetches content from URLs and extracts text using BeautifulSoup."""
    all_text = ""
    scraped_urls = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'}) # Add User-Agent
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text from relevant tags (paragraphs and headings)
            texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            page_text = "\n".join([t.get_text() for t in texts])
            all_text += f"\n\n--- Content from {url} ---\n{page_text}"
            scraped_urls.append(url)
            time.sleep(0.5) # Small delay between requests
        except requests.exceptions.RequestException as e:
            st.warning(f"Could not fetch {url}: {e}")
        except Exception as e:
            st.warning(f"Error processing {url}: {e}")
    
    return all_text.strip(), scraped_urls
# ----------------------------------------------------

# Header
st.title("🌐 Web Search RAG Assistant")
st.markdown("### Ask questions based on web search results")

# Sidebar
with st.sidebar:
    st.header("Search Settings")
    query = st.text_input("Enter your search query:")

    if st.button("Search and Process", type="primary"):
        if query:
            if not SERPER_API_KEY:
                st.error("SERPER_API_KEY not found in environment variables. Please set it in your .env file.")
            else:
                # Check if query has changed
                if query != st.session_state.current_query:
                    reset_chat_history()
                    st.session_state.current_query = query

                with st.spinner("Searching the web and processing content..."):
                    try:
                        # 1. Search the web
                        st.info("Performing web search...")
                        urls = search_web(query, SERPER_API_KEY)
                        if not urls:
                            st.warning("No URLs found for the query.")
                        else:
                            st.info(f"Found {len(urls)} relevant URLs. Fetching content...")
                            # 2. Fetch and extract text
                            extracted_text, scraped_urls = fetch_and_extract_text(urls)
                            if not extracted_text:
                                st.error("Could not extract any text content from the search results.")
                            else:
                                st.info("Processing extracted text for RAG...")
                                # 3. Process text with RAG
                                st.session_state.rag.process_scraped_text(extracted_text)
                                st.session_state.scraped_urls = scraped_urls
                                st.session_state.raw_results = extracted_text
                                st.session_state.query_processed = True
                                st.success("Web content processed successfully!")
                                st.rerun() # Refresh the interface
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a search query")

    st.divider()
    st.markdown("### How to use")
    st.markdown("""
    1. Enter a search query in the input field
    2. Click 'Search and Process'
    3. The system searches the web, extracts content
    4. Ask questions based on the retrieved content
    """)

# --- Show Scraped Details and Raw Results ---
if st.session_state.query_processed:
    st.subheader("Scraped URLs")
    st.write(st.session_state.scraped_urls)

    st.subheader("Raw Extracted Results")
    st.text_area("Raw Results", st.session_state.raw_results, height=300)

# Main chat interface
st.divider()

# Display chat messages
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Chat input
if st.session_state.query_processed:
    question = st.chat_input("Ask a question about the search results...")
    if question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Get answer from RAG
        with st.spinner("Thinking..."):
            try:
                formatted_history = []
                history_to_format = st.session_state.chat_history[:-1]
                for i in range(0, len(history_to_format), 2):
                    if i + 1 < len(history_to_format) and history_to_format[i]["role"] == "user" and history_to_format[i+1]["role"] == "assistant":
                        formatted_history.append((history_to_format[i]["content"], history_to_format[i+1]["content"]))

                answer = st.session_state.rag.ask_question(question, formatted_history)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")
else:
    st.info("👈 Please enter a query and click 'Search and Process' in the sidebar")

# Footer
st.divider()
st.markdown("Built by Vatsa Joshi")
