import os 
from dotenv import load_dotenv # type: ignore
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain, SequentialChain  # Import SequentialChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from sentence_transformers import SentenceTransformer
import textwrap
from langchain_huggingface import HuggingFaceEndpoint
from flask import Flask, request, render_template, jsonify
from langchain_community.vectorstores import Chroma
# for env variables
load_dotenv()

# Use Hugging Face's hosted LLaMA 3 model

app = Flask(__name__)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# embeddings = HuggingFaceEndpoint(
#     repo_id="sentence-transformers/all-MiniLM-L6-v2",
#     task="feature-extraction"
# )


# def create_db_from_youtube_video_url(video_url: str) -> FAISS:  #video_url should be string and the return type of this function is FAISS
#     loader = YoutubeLoader.from_youtube_url(video_url)
#     transcript = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     docs = text_splitter.split_documents(transcript)

#     db = FAISS.from_documents(docs, embeddings)
#     return db

def create_db_from_youtube_video_url(video_url: str):
    loader = YoutubeLoader.from_youtube_url(video_url)
    
    try:
        transcript = loader.load()
        if not transcript: 
            raise ValueError("No transcript available for this video.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)
        db = Chroma.from_documents(docs, embeddings)
        return db

    except Exception as e:
        print(f"Error loading transcript: {e}")
        return None  


def get_response_from_query(db, query, k=4):

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # Hugging Face model repo ID
        task="text-generation",  # Task type
        max_new_tokens=400,  # Limit response length!
        do_sample=False,  # Disable random sampling for deterministic output
    )
    

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_url = request.form["video_url"]
        query = request.form["query"]

        db = create_db_from_youtube_video_url(video_url)

        if db is None:  # If transcript is missing
            return jsonify({"error": "Failed to load transcript. This video may not have captions."})

        response = get_response_from_query(db, query)

        return jsonify({"response": response})

    return render_template("index.html")




if __name__ == "__main__":
    app.run(debug=True)



