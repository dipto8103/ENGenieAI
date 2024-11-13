import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

class RetrievalAndGeneration:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question
            <context>
            {context}
            <context>
            Questions:{input}
            """
        )
        print("Initialized RAG with LLM")

    def create_retrieval_response(self, prompt1, vectors):
        print("Creating retrieval response...")
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start

        print(f"Response time: {response_time} seconds")
        return response['answer'], response["context"]
