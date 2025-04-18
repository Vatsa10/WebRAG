# 🌐 WebRAG Assistant

An intelligent assistant that performs real-time **web search**, **scrapes information**, and allows users to **ask questions** about the fetched content using **Retrieval-Augmented Generation (RAG)** powered by **Groq LLMs**, **LangChain**, and **Serper API**.

Built with  using **Streamlit** and **LangChain**.


---
## 🚀 Features

- 🔎 Web search using Serper (Google-like results)
- 🧠 RAG-based Q&A using Mixtral + DeepSeek models from Groq
- 💬 Conversational memory with LangChain
- 🔗 Contextual vector storage with ChromaDB and HuggingFace embeddings
- 🧼 HTML content cleaning and parsing via BeautifulSoup
- 🖥️ Interactive Streamlit UI

---
### Link to the Output Video: https://drive.google.com/file/d/11PBd-_af_s3qMMHzINCHCOnBJYwlRxQd/view?usp=sharing
---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/WebRAG.git
cd WebRAG
```

### 2. Install Dependencies

Make sure you are using **Python 3.9+** and ideally a virtual environment.

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Example requirements.txt</summary>

```txt
chromadb
langchain
langchain-groq
beautifulsoup4
python-dotenv
streamlit
streamlit-chat
requests
scrapegraphai
nest_asyncio 
crawl4ai
```
</details>

### 3. Set Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
```

---

## 🧪 How to Run

```bash
Remove-Item -Recurse -Force .\chroma_db ( to remove past chromadb data)
streamlit run streamlit_app.py
```

Then open your browser at `http://localhost:8501`.

---

## ✨ How to Use

1. Enter a search query in the sidebar (e.g., "Retrieval Augmented Generation").
2. Click **"Search and Process"**.
3. Wait while the app:
   - Fetches top 5 results via Serper
   - Scrapes the web pages
   - Embeds content and sets up the RAG pipeline
4. Ask questions based on the retrieved information in the chat box!
(you can check the whole process in terminal)

---

## 🧠 Conversational Memory

This assistant uses **LangChain’s ConversationalRetrievalChain**, allowing it to:

- Understand follow-up questions
- Maintain context over multiple turns
- Provide answers that reference previous interactions

Chat history is stored using Streamlit’s `session_state` and passed as structured memory to the LLM.

> ✅ Example:  
> **Q1:** "What is LangChain?"  
> **Q2:** "Can I use it for memory in a chatbot?"  
> The assistant will know Q2 refers to LangChain!

---

## 📁 Project Structure

```
📦 webrag-assistant
┣ 📄 rag_app.py           # Core RAG logic, Groq integration, embeddings
┣ 📄 streamlit_app.py     # Streamlit UI and web search interface
┣ 📄 .env                 # API keys (not committed)
┣ 📁 chroma_db/           # Persistent vector DB for RAG
┗ 📄 README.md            # You're here!
```

---

## 🤖 Models Used

- **Mixtral 8x7B (Groq)** – for web understanding
- **DeepSeek Llama 70B (Groq)** – for response generation
- **BAAI/bge-base-en-v1.5** – embeddings from HuggingFace

---

## 🔐 API Services

| Service     | Purpose         | Free Tier? |
|-------------|------------------|------------|
| [Groq API](https://console.groq.com/)   | Language model inference | ✅ |
| [Serper API](https://serper.dev/) | Google-style web search   | ✅ |

---

## ⚠️ Notes

- ChromaDB is used with persistent storage (`chroma_db/`). It will retain previously processed vectors unless cleared manually.
- All requests are handled synchronously for compatibility.
- The current implementation uses CPU for embedding generation.

---

## 📌 To-Do (Optional Improvements)

- Add chunk metadata filters (e.g., source URLs) in answers.
- Expand support for PDF/Doc/Other sources.
- Add chatbot memory persistence across sessions.
- Add UI toggles for scraping method (Scrapegraph, Crawl4AI, etc.).

---

## 🧑‍💻 Author

**Vatsa Joshi** — [Portfolio](https://vatsa-joshi.vercel.app)

---
