
# Art History Conversational Agent 🎨✨

An interactive **RAG (Retrieval-Augmented Generation)** application that enables users to explore the fascinating world of art history through engaging conversations. Built using **LangChain** and **LangGraph**, and powered by a sleek **Streamlit** interface, this app is your gateway to art history knowledge.

![Example](https://github.com/user-attachments/assets/06d4d239-5b02-4b52-bcee-a9ff359624fc)

---

## ✨ Features

- **Domain-Specific Knowledge:**  The application focuses on art history, with data sourced from the [Boise State Art History Pressbook](https://boisestate.pressbooks.pub/arthistory/).

- **Efficient Data Storage:**  The scraped data is chunked and embedded using a `ingestion.py` script, then stored in a **FAISS (Facebook AI Similarity Search)** database for fast and efficient retrieval.

- **Intelligent Query Handling:**  The agent determines whether a user's question is relevant to art history and processes it accordingly to provide accurate, context-aware responses.

- **Streamlit Interface:**  A user-friendly **Streamlit app** serves as the chatbot interface, maintaining coherent conversations with context-aware memory.

- **Context-Aware Memory:**  Previous chat memory is stored in a **PostgreSQL database**, allowing for dynamic and efficient conversations.


## 🚀 Technologies Used

- **LangChain & LangGraph:** Framework for building and orchestrating the RAG application.
- **FAISS:** Vector store for storing embedded documents.
- **PostgreSQL:** For tracking conversations and maintaining memory.
- **Streamlit:** For a clean, intuitive user interface.


## ⚛️ Agent Workflow

![Workflow](https://github.com/user-attachments/assets/3807b771-fd26-4169-83ea-e370988783a6)


When a user provides a query, the agent processes it through the following steps:

#### **1. Relevance Check:**
   - The agent determines if the query is related to art history.
   - **Non-Relevant Queries:** The agent informs the user that the question is unrelated.
   - **Simple Greetings:** The agent responds using a general-purpose LLM.

####  **2. Context-Aware Query Construction:**
   - Relevant queries are refined into better-constructed questions.
   - Grammar is improved, and chat memory is used to make the query context-aware.

#### **3. Document Retrieval:**
   - The top-5 most relevant documents are retrieved from the FAISS database.

####  **4. Document Grading:**
   - Retrieved documents are graded for relevance.
   - If no documents pass grading:
     - The query is rewritten, and the process is repeated.
     - Wikipedia is used as a supplementary source.
   - If the query cannot be answered, the agent gracefully states, "I don’t know."

####  **5. Answer Generation:**
   - Documents passing the grading step are used to generate an accurate answer.

#### **6. Answer Validation:**
   - The answer is validated to ensure it satisfies the user’s query.
   - Validated answers are saved in memory and presented to the user.


## 🔧 Setup

### → **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

### → **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: `venv\Scripts\activate`
   ```

### → **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### → **Create a `.env` File:**
   ```bash
   # PostgreSQL DB Config
   DB_HOST="<your_host>"
   DB_PORT="<your_port>"
   DB_NAME="<your_database>"
   DB_USER="<your_user>"
   DB_PASSWORD="<your_password>"

   # Google API Key (if required)
   GOOGLE_API_KEY="<your_api_key>"
   ```

### → **Set Up the Database:**
   - Run the `ingestion.py` script to scrape, chunk, and embed the art history data into the FAISS database.
   ```bash
   python ingestion.py
   ```

### → **Run the Streamlit App:**
   ```bash
   streamlit run main.py
   ```


## 🧱 Contributing

Contributions are welcome!  ✨

