import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import openai
import ast

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🏠 제주 부동산 챗봇")

# OpenAI API Key 입력
openai.api_key = ""  # 실제 키 입력

# 데이터 불러오기
@st.cache_resource
def load_data():
    df0 = pd.read_csv("아파트_임베딩.csv")
    df1 = pd.read_csv("오피스텔_임베딩.csv")
    df0["구분"] = "아파트"
    df1["구분"] = "오피스텔"
    df = pd.concat([df0, df1], ignore_index=True)
    df["임베딩"] = df["임베딩"].apply(ast.literal_eval)
    return df

df = load_data()

# NearestNeighbors 모델 준비
@st.cache_resource
def build_nn_model(df):
    X = np.array(df["임베딩"].tolist()).astype("float32")
    model = NearestNeighbors(n_neighbors=5, metric="euclidean")
    model.fit(X)
    return model, X

nn_model, embedding_matrix = build_nn_model(df)

# 유사도 검색 함수
def search_similar(text, top_k=5):
    response = openai.embeddings.create(model="text-embedding-3-small", input=[text])
    query_vec = np.array(response.data[0].embedding).reshape(1, -1).astype("float32")
    distances, indices = nn_model.kneighbors(query_vec, n_neighbors=top_k)
    return df.iloc[indices[0]]

# LLM 응답 생성
def generate_response(contexts, question):
    context_text = "\n---\n".join(contexts)
    prompt = f"""아래는 부동산 데이터 요약입니다.

{context_text}

위 내용을 참고하여 다음 질문에 답변해주세요:  
"{question}"
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


user_input = st.chat_input("궁금한 것을 물어보세요:")

if user_input:
    with st.spinner("검색 중..."):
        top_docs = search_similar(user_input, top_k=5)
        summaries = top_docs["내용"].tolist()
        response = generate_response(summaries, user_input)

    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response)

    with st.expander("🔍 유사 단지 정보 보기"):
        st.dataframe(top_docs[["단지명", "구분"]])
