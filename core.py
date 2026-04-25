from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

# Initialize conversation history
conversation_history = []

# PDF text extraction function
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create or load vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Retrieve top-k most relevant chunks
def get_relevant_context(user_question, top_k=5):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=top_k)  # Retrieve top-k relevant chunks
    return docs

# Format context from retrieved chunks
def format_context(docs):
    return " ".join([doc.page_content for doc in docs])

# Manage conversation history with a sliding window approach
def manage_conversation_history(user_question, bot_response, max_turns=3):
    global conversation_history
    if len(conversation_history) >= max_turns * 2:
        conversation_history = conversation_history[2:]  # Remove oldest interaction
    conversation_history.extend([f"User: {user_question}", f"Bot: {bot_response}"])
    return "\n".join(conversation_history)

# Define a prompt template with structured instructions
def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant. Read the context carefully, and answer as if you're explaining to a person in a simple and friendly manner.
    Use plain language, avoid technical jargon, and provide a concise response. Only structure your answer in points if there are multiple distinct points or steps to explain.

    <context>
    {context}
    </context>
    Question: {question}

    Your answer should be:
    - Friendly and conversational
    - In complete sentences if the answer is short or straightforward
    - Structured as a list only if there are multiple points or steps
    - Accurate and based only on the context provided
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = create_stuff_documents_chain(model, prompt)
    return chain

# Handle user input and generate response
def user_input(user_question):
    # Simplify the question to improve comprehension
    simplified_question = simplify_question(user_question)
    
    # Retrieve relevant context from FAISS
    docs = get_relevant_context(simplified_question, top_k=5)
    context = format_context(docs)
    
    # Generate response using the conversational chain
    chain = get_conversational_chain()
    response = chain.invoke({"context": docs, "question": simplified_question})
    
    # Manage conversation history
    bot_response = response
    conversation_history_text = manage_conversation_history(simplified_question, bot_response)
    
    return bot_response

def simplify_question(question):
    # Basic preprocessing steps
    question = question.lower().strip()  # Lowercase and trim whitespace
    question = question.replace("please", "").replace("could you", "").replace("?", "")
    # More advanced NLP techniques could be applied here, such as paraphrasing
    return question
# TSR