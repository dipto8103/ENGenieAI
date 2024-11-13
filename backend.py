import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

# Load environment variables
#groq_api_key = os.getenv("GROQ_API_KEY")
#google_api_key = os.getenv("GOOGLE_API_KEY")

class DocumentHandler:
    def __init__(self):
        groq_api_key = "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" 
        google_api_key = "AIzaSyDtsDJuAg0vMw6js7Iw_d2l45e2JVy_HxI"
        self.groq_api_key = groq_api_key
        os.environ["GOOGLE_API_KEY"] = google_api_key

    def load_llm(self):
        print("Loading LLM...")
        return ChatGroq(groq_api_key=self.groq_api_key, model_name="llama-3.2-11b-vision-preview")

    def create_embeddings(self):
        print("Creating embeddings...")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def load_documents(self, uploaded_files):
        print("Loading documents...")
        docs = []
        if not os.path.exists("temp"):
            os.makedirs("temp")
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyMuPDFLoader(file_path=file_path)
            docs.extend(loader.load())
        print(f"Loaded {len(docs)} documents")
        return docs

    def create_vector_store(self, documents, embeddings):
        print("Creating vector store...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)
        vectors = FAISS.from_documents(final_documents, embeddings)
        print("Vector store created")
        return vectors

    def normal_chat(self, prompt, context):
        messages = []
        for msg in context:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            else:
                messages.append(AIMessage(content=msg['content']))
        messages.append(HumanMessage(content=prompt))
        llm = self.load_llm()

        # Directly call the LLM with the constructed messages
        response = llm(messages)
        return response.content

    def socratic_chat(self, prompt, context):
        socratic_prompt_template = """
        You are an AI bot, playing the role of a Socratic tutor with expertise in all engineering subject domains, catering to the diverse needs of engineering students.

        Let's explore a core concept in engineering. Consider why [specific topic] might be essential for [engineering domain]. Upon reflecting on your answer, we will further delve into the complexities of this concept, discussing its implications and applications in greater detail.

        As a Socratic tutor, remember your responsibility is not to simply supply answers, but rather to facilitate my learning through purposeful questioning. Your role is to stimulate my critical thinking, guiding me towards a deep understanding of the relationships and dynamics within [engineering domain], and how they interconnect with other fields.

        As we progress, ask me to reflect on how [related sub-topic or application] can aid in enhancing our understanding and utilization of [specific topic].

        Your approach should be patient, adaptable, and stimulating, continuously encouraging me to think, analyze, and connect these concepts to create my own holistic understanding of key topics in engineering. Do not hallucinate in between topics and keeo engaging the topic in a forward way.
        """

        prompt_text = socratic_prompt_template.replace("[specific topic]", prompt).replace("[engineering domain]", "engineering")

        messages = []
        for msg in context:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            else:
                messages.append(AIMessage(content=msg['content']))
        messages.append(HumanMessage(content=prompt_text))
        
        llm = self.load_llm()

        # Directly call the LLM with the constructed messages
        response = llm(messages)
        return response.content
