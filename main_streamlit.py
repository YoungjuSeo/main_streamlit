import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler

# 0. .env 파일에서 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 1. 스트리밍 처리를 위한 핸들러
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# 2. PDF 로드 및 문서 분할 함수
def pdf_to_document(uploaded_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
    return pages

# --- Streamlit UI ---
st.set_page_config(page_title="ChatPDF 📮", layout="centered")
st.title("ChatPDF 📮")
st.write("---")

# 파일 업로드 필드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])

# API 키가 설정되어 있고 파일이 업로드된 경우 실행
if uploaded_file:
    if not api_key:
        st.error(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    else:
        # 문서 처리 및 벡터 DB 구축
        with st.spinner("문서를 분석 중입니다..."):
            pages = pdf_to_document(uploaded_file)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(pages)
            
            embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
            vector_db = Chroma.from_documents(documents=texts, embedding=embeddings_model)
            retriever = vector_db.as_retriever()

        st.success("분석 완료!")
        st.header("PDF에게 질문하기")
        question = st.text_input("질문 내용")

        if st.button("질문하기"):
            if question:
                chat_box = st.empty()
                stream_handler = StreamHandler(chat_box)
                
                llm = ChatOpenAI(
                    model="gpt-4o-mini", 
                    temperature=0, 
                    streaming=True, 
                    callbacks=[stream_handler]
                )

                template = """You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Keep the answer concise.
                
                Question: {question} 
                Context: {context} 
                Answer:"""
                prompt = ChatPromptTemplate.from_template(template)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                with st.spinner("답변 생성 중..."):
                    rag_chain.invoke(question)