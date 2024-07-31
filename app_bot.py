import os                                                         # para uso da chave API
import streamlit as st                                            # para a aplicação/interface
from PIL import Image                                             # imagem da aplicação
from langchain_community.document_loaders import PyPDFLoader      # para carregar o PDF
from langchain_huggingface import HuggingFaceEmbeddings           # para gerar os embeddings
from langchain import FAISS                                       # para a vector store
from langchain_groq import ChatGroq                               # para usar o LLM


# Configuração da chave da API do Groq
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise ValueError("A chave da API não está definida")
os.environ["GROQ_API_KEY"] = api_key

# Lista com os nomes dos arquivos PDF divididos
pdf_files = [f"pdfs/resolucao_GR-029-2024-{i}.pdf" for i in range(1, 12)]

# Carregar e processar todos os PDFs divididos
@st.cache_data
def process_pdfs(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load_and_split()
        documents.extend(pages)
    return documents

# Cria a vector store a partir dos documentos processados
@st.cache_data
def create_vector_store(_documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(_documents, embeddings)
    return db

# Consulta ao modelo Groq usando RAG
@st.cache_data
def query_groq_rag(query, _db):
    docs = _db.similarity_search(query)
    content = "\n".join([x.page_content for x in docs])
    qa_prompt = "Use o contexto seguinte para responder à pergunta. Se não souber a resposta, diga que não possui essa informação."
    input_text = qa_prompt + "\nContexto:" + content + "\nPergunta:\n" + query
    llm = ChatGroq(model="llama3-70b-8192")
    result = llm.invoke(input_text)
    return result.content


# Função principal para uso com Streamlit
def main():
    st.set_page_config(page_title = 'Vestibular Unicamp 2025 Chatbot', \
        page_icon = 'img/icon.png',
        #layout="wide",
        initial_sidebar_state='expanded'
    )

    st.title("Vestibular Unicamp 2025")
    st.markdown("---")
    st.write("Tem alguma dúvida? Digite sua pergunta abaixo:")

    # Apresenta a imagem na barra lateral da aplicação
    image = Image.open("img/side-image.png")
    st.sidebar.image(image)

    # Criar a base de dados a partir do documento
    documents = process_pdfs(pdf_files)
    db = create_vector_store(documents)

    # Histórico de conversas estilo chat-gpt
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Digite sua mensagem e pressione Enter"):
        # Adiciona a mensagem do usuario no historico
        st.session_state.messages.append({"role": "user", "content": query})
        # Exibe a mensagem do usuario
        with st.chat_message("user"):            
            st.markdown(query)
        with st.spinner("Processando..."):
            answer = query_groq_rag(query, db)

        st.session_state.messages.append({"role": "assistant", "content": answer})    
        # Exibe a mensagem do assistente
        with st.chat_message("assistant"):            
            st.markdown(answer)
        

        
            
    
# Executa a função principal
if __name__ == "__main__":
    main()
