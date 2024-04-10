
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from collections import Counter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader

def get_chatbot_response(user_input):
    loader = CSVLoader("TXTCombined.csv", encoding='utf-8')
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings_model)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
    result = qa_chain({"query": user_input})
    return result["result"]


def main():

  #chat
  if "messages" not in st.session_state:
      st.session_state.messages = []

  #초기화 버튼
  if 'check_reset' not in st.session_state:
    st.session_state['check_reset'] = False

  #페이지 기본 설정
  st.set_page_config(page_title = '챗봇', layout = 'wide')

  #제목
  st.header("안녕하세요 고객센터입니다")
  st.markdown('---')


  # React to user input
  if user_input := st.chat_input("궁금한 것을 물어보세요!"):
      
      # Display user message in chat message container
      with st.chat_message("user"):
          st.markdown(user_input)

      # Add user message to chat history
      st.session_state.messages.append({"role": "user", "content": user_input})

      response = get_chatbot_response(user_input)

      # Display assistant response in chat message container
      with st.chat_message("assistant"):
          st.markdown(response, unsafe_allow_html=True)

      # Add assistant response to chat history
      st.session_state.messages.append({"role": "assistant", "content": response})

if __name__=='__main__':
  main()
