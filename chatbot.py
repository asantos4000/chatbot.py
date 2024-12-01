import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub

# Configuración de la página
st.set_page_config(page_title="ChatBot Demo", page_icon="🤖")

def init_chatbot(text_content):
    # Dividir el texto en chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text_content)

    # Crear embeddings
    embeddings = HuggingFaceEmbeddings()

    # Crear base de datos vectorial
    vectorstore = Chroma.from_texts(chunks, embeddings)

    # Configurar memoria
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Crear cadena de conversación
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=HuggingFaceHub(repo_id="google/flan-t5-base"),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return qa_chain

# Interfaz de usuario
st.title("ChatBot Demo 🤖")

# Área para cargar el texto base
if "text_content" not in st.session_state:
    st.session_state.text_content = ""

text_input = st.text_area("Ingresa el texto base para el chatbot:", height=200)
if st.button("Inicializar Chatbot"):
    st.session_state.text_content = text_input
    st.session_state.chatbot = init_chatbot(text_input)
    st.success("¡Chatbot inicializado!")

# Área de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" in st.session_state:
    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("Escribe tu mensaje aquí"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Respuesta del chatbot
        with st.chat_message("assistant"):
            response = st.session_state.chatbot({"question": prompt})
            st.markdown(response['answer'])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

# Created/Modified files during execution:
print("chatbot.py")
