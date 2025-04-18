import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import chromadb
from chromadb.config import Settings
import nest_asyncio
import sys
import asyncio

# Load environment variables
load_dotenv()
print("[INFO] Loaded environment variables.")

# Set Windows event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

# ChromaDB settings
persist_directory = "chroma_db"
chroma_setting = Settings(anonymized_telemetry=False)
client = chromadb.PersistentClient(path=persist_directory, settings=chroma_setting)

class WebRAG:
    def __init__(self):
        print("[INIT] Initializing WebRAG...")

        # LLMs
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768"
        )
        self.response_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="DeepSeek-R1-Distill-Llama-70B",
            temperature=0.6,
            max_tokens=2048,
        )

        # Embeddings
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,          # Reduced chunk size for more granularity
            chunk_overlap=100
        )

        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            client=client,
            persist_directory=persist_directory,
            client_settings=chroma_setting
        )

        self.qa_chain = None
        print("[INIT] WebRAG initialized successfully.")

    def process_scraped_text(self, text_content):
        """Processes raw text from web scraping into vector embeddings and builds the QA chain."""
        try:
            if not text_content:
                raise ValueError("No text content provided for processing.")

            print("[PROCESS] Length of scraped text:", len(text_content))

            text_content = text_content.encode('utf-8', errors='ignore').decode('utf-8')

            doc = Document(page_content=text_content, metadata={"source": "web_search"})
            chunks = self.text_splitter.split_documents([doc])

            print(f"[PROCESS] ✅ Created {len(chunks)} chunks from web search text.")

            if not chunks:
                raise ValueError("Text content could not be split into chunks.")

            # Store chunks in ChromaDB in batches
            MAX_BATCH_SIZE = 100
            for i in range(0, len(chunks), MAX_BATCH_SIZE):
                i_end = min(len(chunks), i + MAX_BATCH_SIZE)
                batch = chunks[i:i_end]
                self.vector_store.add_documents(batch)
                print(f"[VECTOR] 📦 Stored vectors for batch {i} to {i_end} (Total: {len(batch)})")

            # Build the QA chain with top-k retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.response_llm,
                retriever=retriever,
                return_source_documents=True
            )
            print("[CHAIN] ✅ QA chain created successfully.")

        except Exception as e:
            print(f"[ERROR] Error processing scraped text: {str(e)}")
            raise Exception(f"Error processing scraped text: {str(e)}")

    def ask_question(self, question, chat_history=[]):
        """Handles user queries using the initialized QA chain."""
        try:
            if not self.qa_chain:
                raise ValueError("No QA chain found. Please process content first.")

            response = self.qa_chain.invoke({
                "question": question,
                "chat_history": chat_history[:4000]
            })

            raw_answer = response.get("answer", "Sorry, I couldn't generate an answer.")
            final_answer = raw_answer.strip()

            # Extract better if </think> used in Groq responses
            if isinstance(final_answer, str) and "</think>" in final_answer:
                parts = final_answer.split("</think>", 1)
                if len(parts) > 1:
                    final_answer = parts[1].strip()

            if final_answer.startswith("Answer:"):
                final_answer = final_answer[len("Answer:"):].strip()

            print("[QA] Answer generated:", final_answer[:200], "...")
            return final_answer

        except Exception as e:
            raise Exception(f"[ERROR] Could not generate answer: {str(e)}")


# Optional CLI testing
def main():
    rag = WebRAG()
    print("[MAIN] CLI Demo - Start")

    # Placeholder for manual text input or scraping method
    text = input("Paste some text or simulate a scraped result: ")
    rag.process_scraped_text(text)

    chat_history = []
    while True:
        question = input("\nYour question (type 'quit' to exit): ")
        if question.lower() == "quit":
            break
        try:
            answer = rag.ask_question(question, chat_history)
            print("\nAnswer:", answer)
            chat_history.append((question, answer))
        except Exception as e:
            print(str(e))
            break


if __name__ == "__main__":
    main()
