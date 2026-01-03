import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 0. API 키 로드
load_dotenv()

# 1. PDF 문서 로드
print("1. 문서를 읽어오는 중입니다 (OpenAI)...")
loader = PyPDFLoader('dong.pdf')
pages = loader.load_and_split()

# 2. 텍스트 분할
print("2. 문서를 쪼개는 중입니다...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(pages)

# 3. 임베딩 & 4. 벡터 저장소 구축
print("3. OpenAI 임베딩으로 벡터 DB 구축 중...")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
vector_db = Chroma.from_documents(documents=texts, embedding=embeddings_model)

# 5. 검색기 및 생성기 설정
print("4. GPT-4o-mini 모델 연결 중...")
retriever = vector_db.as_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 6. 프롬프트 설정 (에러 방지를 위해 직접 작성!)
# 원래 hub에서 가져오려던 내용을 직접 정의합니다.
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. RAG 체인 구성
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. 질문 실행
print("-" * 50)
question = "점순이가 주인공에게 느끼는 감정은 무엇이야?"
print(f"질문: {question}")

try:
    print("AI가 답변 생성 중...")
    result = rag_chain.invoke(question)
    print("-" * 50)
    print(f"최종 답변:\n{result}")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
    