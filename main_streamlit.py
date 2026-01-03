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

# 0. 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 1. 스트리밍 핸들러 정의
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# 2. PDF 문서 로드 및 분할 함수
def pdf_to_document(uploaded_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
    return pages

# --- Streamlit UI 설정 ---
st.set_page_config(page_title="ChatPDF Bot 🤖", layout="centered")
st.title("ChatPDF Bot 🤖")
st.write("---")

# 3. 세션 상태(대화 내역) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. 파일 업로드 및 데이터 처리 (최초 1회만 수행되도록 세션 활용)
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])

if uploaded_file:
    if "retriever" not in st.session_state:
        with st.spinner("문서를 분석 중입니다..."):
            pages = pdf_to_document(uploaded_file)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(pages)
            
            embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
            vector_db = Chroma.from_documents(documents=texts, embedding=embeddings_model)
            st.session_state.retriever = vector_db.as_retriever()
            st.success("분석 완료! 대화를 시작하세요.")

    # 5. 기존 대화 내역 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 6. 채팅 입력 및 답변 생성
    if prompt := st.chat_input("PDF 내용에 대해 질문하세요"):
        # 사용자 메시지 표시 및 저장
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # AI 답변 생성 공간 확보
        with st.chat_message("assistant"):
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
            prompt_template = ChatPromptTemplate.from_template(template)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # RAG 체인 구성
            rag_chain = (
                {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            # 답변 실행
            full_response = rag_chain.invoke(prompt)
            st.session_state.messages.append({"role": "assistant", "content": full_response})