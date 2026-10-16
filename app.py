"""
Consultor Virtual Especialista em Seguros com RAG
Desenvolvido em Streamlit com integração Microsoft Foundry / Azure AI
"""

import io
import os
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# ---------------------------------------------------------
# Configurações do Microsoft Foundry / Azure AI
# ---------------------------------------------------------
DEFAULT_AZURE_ENDPOINT = "https://marcelomaffeis-05082026-resource.services.ai.azure.com/"
DEFAULT_AZURE_API_KEY = "6t2pTNsmNafDyo53MDAFIrvvQRBZXP7dLcr5j3TLxMS34LQVq3ZUJQQJ99CHACHYHv6XJ3w3AAAAACOGT7os"
DEFAULT_AZURE_MODEL = "gpt-4.1-mini"
AZURE_API_VERSION = "2024-06-01"

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
    """Divide o texto em blocos menores preservando sobreposição contextual."""
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
    """Recupera os blocos mais similares à pergunta colocada."""
    if "rag_chunks" not in st.session_state or not st.session_state["rag_chunks"]:
        return []

    embedder = load_embedding_model()
    query_vector = embedder.encode([query])
    doc_vectors = st.session_state["rag_embeddings"]

    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    relevant = [st.session_state["rag_chunks"][idx] for idx in top_indices if similarities[idx] > 0.15]
    return relevant


# ---------------------------------------------------------
# Engenharia de Prompt (System Instruction)
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "Você é o 'SegurAI', um consultor técnico e subscritor sénior especialista no mercado segurador brasileiro e internacional.\n"
    "A sua missão é apoiar corretores, analistas corporativos, gestores de risco e segurados em dúvidas técnicas, operacionais e conceituais de seguros.\n\n"
    "Diretrizes RAG:\n"
    "- Quando forem fornecidos 'Trechos do Documento Carregado', dê prioridade máxima aos factos, cláusulas e números contidos neles.\n"
    "- Se a resposta constar expressamente no documento (como manuais SUSEP, apólices, condições gerais), cite as regras ou parâmetros encontrados.\n"
    "- Se o documento não cobrir o assunto em questão, complemente tecnicamente com o seu conhecimento padrão de mercado (SUSEP, boas práticas, franquias, POS, etc.), alertando claramente o utilizador de que aquela informação advém da prática geral e não do documento carregado.\n\n"
    "Tom de voz:\n"
    "- Profissional, analítico, estruturado e objetivo em Português."
)

# ---------------------------------------------------------
# Barra Lateral (Sidebar)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🛡️ SegurAI Analytics")
    st.caption("Microsoft Foundry / Azure AI Edition")
    st.markdown("---")

    # Autenticação e Configuração
    st.subheader("⚙️ Configurações do Foundry")
    api_endpoint = st.text_input("Endpoint:", value=DEFAULT_AZURE_ENDPOINT)
    api_key = st.text_input("API Key:", value=DEFAULT_AZURE_API_KEY, type="password")
    model_deployment = st.text_input("Modelo / Deployment:", value=DEFAULT_AZURE_MODEL)

    st.markdown("---")

    # Módulo RAG: Upload de Documentos
    st.subheader("📄 Base Documental (RAG)")
    uploaded_files = st.file_uploader(
        "Carregue Apólices, Manuais ou Relatórios:",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        help="Permite ao assistente consultar e citar cláusulas do documento.",
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
        if st.button("Remover Documento da Memória", use_container_width=True):
            st.session_state.pop("rag_chunks", None)
            st.session_state.pop("rag_embeddings", None)
            st.session_state["doc_loaded"] = False
            st.rerun()

    st.markdown("---")

    # Sugestões Rápidas
    st.subheader("💡 Sugestões Rápidas")
    sugestoes = [
        "Qual é o prazo regulamentar de liquidação de sinistros segundo a SUSEP?",
        "Qual a diferença entre LMI e LMG segundo o manual?",
        "O que diz o documento sobre a suspensão de prazo de regulação?",
        "Quais as garantias típicas de Riscos de Engenharia?",
    ]
    for s in sugestoes:
        if st.button(s, use_container_width=True):
            st.session_state["pending_prompt"] = s

    st.markdown("---")
    if st.button("🧹 Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Histórico de Sessão
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Olá! Sou o seu Consultor Virtual de Seguros alimentado pelo modelo "
                f"**{model_deployment}** via Microsoft Foundry. Carregue as suas apólices ou manuais "
                "para conversarmos com suporte RAG!"
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

prompt_input = st.chat_input("Coloque a sua dúvida sobre o mercado ou sobre o manual indexado...")
if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    active_prompt = st.session_state.pop("pending_prompt")
else:
    active_prompt = prompt_input

# ---------------------------------------------------------
# Processamento com AzureOpenAI + Streaming
# ---------------------------------------------------------
if active_prompt:
    if not api_key or not api_endpoint:
        st.warning("⚠️ Forneça o Endpoint e a API Key na barra lateral para prosseguir.")
    else:
        # 1. Regista pergunta do utilizador
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_prompt)

        # 2. Recupera contexto via RAG
        relevant_chunks = retrieve_relevant_chunks(active_prompt, top_k=3)
        context_block = ""
        if relevant_chunks:
            context_block = "\n\n--- [TRECHOS RELEVANTES DOS DOCUMENTOS CARREGADOS] ---\n"
            for i, chunk in enumerate(relevant_chunks, start=1):
                context_block += f"[Bloco {i}]:\n{chunk}\n\n"
            context_block += "--- [FIM DO CONTEXTO DOCUMENTAL] ---\n"

        # 3. Monta mensagens no formato padrão OpenAI / Azure
        messages_payload = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

        for m in st.session_state.messages[:-1]:
            messages_payload.append({"role": m["role"], "content": m["content"]})

        final_prompt = active_prompt
        if context_block:
            final_prompt = (
                f"{active_prompt}\n\n"
                f"Considere os seguintes trechos retirados dos ficheiros carregados para fundamentar a resposta:\n"
                f"{context_block}"
            )

        messages_payload.append({"role": "user", "content": final_prompt})

        # 4. Chamada com streaming ao modelo do Microsoft Foundry
        with st.chat_message("assistant", avatar="🛡️"):
            try:
                # Sanitiza o endpoint (remove eventuais aspas ou barras finais duplicadas)
                clean_endpoint = api_endpoint.strip().rstrip('"').rstrip("/")

                client = AzureOpenAI(
                    azure_endpoint=clean_endpoint,
                    api_key=api_key.strip(),
                    api_version=AZURE_API_VERSION,
                )

                response_stream = client.chat.completions.create(
                    model=model_deployment.strip(),
                    messages=messages_payload,
                    temperature=0.2,
                    stream=True,
                )

                def stream_generator():
                    for chunk in response_stream:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                yield delta

                full_response = st.write_stream(stream_generator())
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Erro ao comunicar com o Microsoft Foundry: {str(e)}")
