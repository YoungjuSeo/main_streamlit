import streamlit as st

# 1. 페이지 설정
st.set_page_config(page_title="PDF 챗봇", layout="centered")
st.header("PDF 챗봇 서비스 🤖")

# 2. 대화 히스토리 초기화 (메모리 생성)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 기존 대화 내역을 화면에 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. 사용자 질문 입력 및 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 화면 출력 및 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI 답변 생성 및 화면 출력 (여기서는 예시 답변 사용)
    with st.chat_message("assistant"):
        response = f"질문에 대한 답변입니다: {prompt}" # 실제로는 여기에 RAG 로직이 들어감
        st.markdown(response)
    
    # AI 메시지 저장
    st.session_state.messages.append({"role": "assistant", "content": response})