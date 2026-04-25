import os
from google import genai
from dotenv import load_dotenv

from core import user_input, get_pdf_text, get_text_chunks, get_vector_store
    
# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def main():

    pdf_docs = ["cv.pdf"] # Replace with your actual PDF file path

    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    user_question = "What is the name of the person in the CV?" # Replace with your actual question

    return user_input(user_question)


if __name__ == "__main__":
    output = main()
    print(output)