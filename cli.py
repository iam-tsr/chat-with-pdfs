from core import user_input, get_pdf_text, get_text_chunks, get_vector_store

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