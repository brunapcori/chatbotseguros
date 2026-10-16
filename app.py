"""
Consultor Virtual Especialista em Seguros com RAG
Licença MIT - Streamlit + Google GenAI + RAG Local
"""

import io
import os
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Configuração da Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Consultor de Seguros AI (RAG)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Modelo de Embeddings Local (Cache para performance)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="A carregar modelo de embeddings local...")
def load_embedding_model():
    # Modelo multilíngue leve e otimizado para CPU
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# ---------------------------------------------------------
# Funções de Leitura e Fragmentação de Documentos
# ---------------------------------------------------------
def extract_text_from_file(uploaded_file) -> str:
    """Extrai texto de PDF, TXT ou CSV."""
    text = ""
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif filename.endswith(".txt") or filename.endswith(".csv"):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Erro ao processar o ficheiro {uploaded_file.name}: {e}")
    return text


def split_text_into_chunks(text: str, chunk_size: int = 700, chunk_overlap: int = 100) -> list[str]:
    """Divide texto em blocos menores preservando sobreposição contextual."""
    if not text.strip():
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - chunk_overlap
    return chunks


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list[str]:
    """Recupera os blocos mais similares à pergunta do utilizador."""
    if "rag_chunks" not in st.session_state or not st.session_state["rag_chunks"]:
        return []

    embedder = load_embedding_model()
    query_vector = embedder.encode([query])
    doc_vectors = st.session_state["rag_embeddings"]

    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Retorna apenas blocos com relevância aceitável
    relevant = [st.session_state["rag_chunks"][idx] for idx in top_indices if similarities[idx] > 0.2]
    return relevant


# ---------------------------------------------------------
# Engenharia de Prompt (System Instruction)
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "Você é o 'SegurAI', um consultor técnico e subscritor sênior especialista no mercado segurador brasileiro e internacional.\n"
    "Sua missão é auxiliar corretores, analistas corporativos, gestores de risco e segurados em dúvidas técnicas, operacionais e conceituais de seguros.\n\n"
    "Diretrizes RAG:\n"
    "- Quando forem fornecidos 'Trechos do Documento Carregado', dê prioridade máxima aos factos e cláusulas contidos neles.\n"
    "- Se a resposta constar expressamente no documento, cite as regras ou parâmetros encontrados.\n"
    "- Se o documento não cobrir o assunto em questão, complemente tecnicamente com o seu conhecimento padrão de mercado (SUSEP, boas práticas, franquias, POS, etc.), alertando claramente o utilizador de que aquela informação advém da prática geral e não do ficheiro carregado.\n\n"
    "Tom de voz:\n"
    "- Profissional, analítico, estruturado e objetivo em Português."
)

# ---------------------------------------------------------
# Obtenção da API Key
# ---------------------------------------------------------
def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    return api_key


api_key_default = get_api_key()

# ---------------------------------------------------------
# Barra Lateral (Sidebar) com Upload de Ficheiro RAG
# ---------------------------------------------------------
with st.sidebar:
    st.title("🛡️ SegurAI Analytics")
    st.caption("Assistente Técnico de Riscos & RAG")
    st.markdown("---")

    # Autenticação
    if not api_key_default:
        st.subheader("🔑 Autenticação")
        user_key = st.text_input(
            "Gemini API Key:",
            type="password",
            help="Insira a sua chave do Google AI Studio.",
        )
        api_key_to_use = user_key.strip()
    else:
        api_key_to_use = api_key_default
        st.success("API Key detetada!")

    st.markdown("---")

    # Módulo RAG: Upload de Documentos
    st.subheader("📄 Base Documental (RAG)")
    uploaded_files = st.file_uploader(
        "Carregue Apólices, Condições Gerais ou Relatórios:",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        help="Permite ao assistente consultar e citar cláusulas do seu próprio documento.",
    )

    if st.button("Indexar Ficheiro(s)", use_container_width=True, type="primary"):
        if uploaded_files:
            with st.spinner("A indexar documentos..."):
                all_text = ""
                for up_file in uploaded_files:
                    all_text += extract_text_from_file(up_file) + "\n"

                chunks = split_text_into_chunks(all_text)
                if chunks:
                    embedder = load_embedding_model()
                    embeddings = embedder.encode(chunks, show_progress_bar=False)

                    st.session_state["rag_chunks"] = chunks
                    st.session_state["rag_embeddings"] = embeddings
                    st.session_state["doc_loaded"] = True
                    st.success(f"{len(chunks)} blocos indexados com sucesso!")
                else:
                    st.warning("Não foi possível extrair texto dos ficheiros enviados.")
        else:
            st.info("Selecione pelo menos um ficheiro antes de indexar.")

    if st.session_state.get("doc_loaded"):
        st.info("🟢 Documento ativo na memória RAG.")
        if st.button("Remover Documento da Memória"):
            st.session_state.pop("rag_chunks", None)
            st.session_state.pop("rag_embeddings", None)
            st.session_state["doc_loaded"] = False
            st.rerun()

    st.markdown("---")

    # Tópicos Rápidos
    st.subheader("💡 Sugestões Rápidas")
    sugestoes = [
        "Resuma as principais exclusões do documento carregado",
        "Qual é o valor da franquia ou POS indicado na apólice?",
        "Diferença entre D&O e E&O na prática",
        "Como funciona a regulação de um sinistro?",
    ]
    for s in sugestoes:
        if st.button(s, use_container_width=True):
            st.session_state["pending_prompt"] = s

    st.markdown("---")
    if st.button("🧹 Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Inicialização de Sessão
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Olá! Sou o seu Consultor Virtual de Seguros. Pode colocar questões conceituais "
                "ou carregar as suas apólices e relatórios em PDF na barra lateral para analisarmos cláusulas específicas juntos."
            ),
        }
    ]

# ---------------------------------------------------------
# Interface Principal de Chat
# ---------------------------------------------------------
st.title("Consultor Virtual Especialista em Seguros")
st.markdown("Análise técnica de termos de mercado, gestão de sinistros e consulta a documentos indexados via **RAG**.")

for msg in st.session_state.messages:
    avatar = "🛡️" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

prompt_input = st.chat_input("Coloque a sua dúvida sobre o mercado ou sobre o ficheiro carregado...")
if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    active_prompt = st.session_state.pop("pending_prompt")
else:
    active_prompt = prompt_input

# ---------------------------------------------------------
# Processamento com Recuperação RAG + Streaming
# ---------------------------------------------------------
if active_prompt:
    if not api_key_to_use:
        st.error("⚠️ Insira a chave da API Gemini na barra lateral ou em secrets para prosseguir.")
    else:
        # 1. Regista mensagem do utilizador
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_prompt)

        # 2. Busca de contexto via RAG (se houver documento ativo)
        relevant_chunks = retrieve_relevant_chunks(active_prompt, top_k=3)
        context_block = ""
        if relevant_chunks:
            context_block = "\n\n--- [TRECHOS RELEVANTES DOS DOCUMENTOS CARREGADOS] ---\n"
            for i, chunk in enumerate(relevant_chunks, start=1):
                context_block += f"[Bloco {i}]:\n{chunk}\n\n"
            context_block += "--- [FIM DO CONTEXTO DOCUMENTAL] ---\n"

        # 3. Monta histórico estruturado para o SDK
        contents_payload = []
        for m in st.session_state.messages[:-1]:
            role_tag = "user" if m["role"] == "user" else "model"
            contents_payload.append(
                types.Content(
                    role=role_tag,
                    parts=[types.Part.from_text(text=m["content"])],
                )
            )

        # Na mensagem atual do utilizador, injeta o contexto recuperado
        final_user_content = active_prompt
        if context_block:
            final_user_content = (
                f"{active_prompt}\n\n"
                f"Considere os seguintes trechos retirados dos ficheiros carregados para fundamentar a resposta:\n"
                f"{context_block}"
            )

        contents_payload.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=final_user_content)],
            )
        )

        # 4. Geração com Streaming
        with st.chat_message("assistant", avatar="🛡️"):
            try:
                client = genai.Client(api_key=api_key_to_use)

                response
