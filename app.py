"""
Sistema de Inteligência Regulatória e Técnica de Seguros
Padrão Visual Institucional / SUSEP - Integrado ao Microsoft Foundry
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
# Configuração da Página e Layout
# ---------------------------------------------------------
st.set_page_config(
    page_title="Portal de Análise Técnica de Seguros | Padrão SUSEP",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Estilização CSS Corporativa (Inspirada no Padrão Digital Gov / SUSEP)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Tipografia e Fundo */
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Barra Superior Institucional */
    .gov-header {
        background-color: #071D41;
        padding: 18px 24px;
        border-radius: 4px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 3px solid #FFCD07;
    }
    .gov-title {
        color: #FFFFFF;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin: 0;
    }
    .gov-subtitle {
        color: #A6B4C9;
        font-size: 0.82rem;
        margin-top: 3px;
    }
    .gov-badge {
        background-color: #0C326F;
        color: #E2E8F0;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 3px;
        border: 1px solid #1E4E98;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Cards e Caixas Informativas */
    .corp-card {
        background-color: #F8FAFC;
        border-left: 4px solid #071D41;
        border-radius: 0 4px 4px 0;
        padding: 14px 18px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        color: #334155;
        font-size: 0.9rem;
    }
    
    .badge-rag-active {
        display: inline-flex;
        align-items: center;
        background-color: #E6F4EA;
        color: #137333;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #CEEAD6;
    }

    /* Ajuste de Botões para Estilo Corporativo */
    div.stButton > button:first-child {
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.2s ease-in-out;
    }

    /* Linha divisória suave */
    hr {
        margin: 1rem 0;
        border-color: #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Modelo de Embeddings Local (Cache para performance)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo vetorial de conformidade...")
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# ---------------------------------------------------------
# Processamento de Documentos
# ---------------------------------------------------------
def extract_text_from_file(uploaded_file) -> str:
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
        st.error(f"Falha ao ler o arquivo {uploaded_file.name}: {e}")
    return text


def split_text_into_chunks(text: str, chunk_size: int = 700, chunk_overlap: int = 100) -> list[str]:
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
    if "rag_chunks" not in st.session_state or not st.session_state["rag_chunks"]:
        return []

    embedder = load_embedding_model()
    query_vector = embedder.encode([query])
    doc_vectors = st.session_state["rag_embeddings"]

    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [st.session_state["rag_chunks"][idx] for idx in top_indices if similarities[idx] > 0.15]


# ---------------------------------------------------------
# Instrução do Sistema (Diretrizes Regulatórias)
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "Você atua como um Consultor Técnico e Regulatório Sênior de Seguros, fundamentado no arcabouço normativo "
    "do Sistema Nacional de Seguros Privados (SNSP), nas Resoluções CNSP, Circulares da SUSEP e no Código Civil Brasileiro.\n\n"
    "Princípios de Resposta:\n"
    "1. Fidelidade Documental (RAG):\n"
    "   - Havendo 'Trechos dos Documentos Indexados', priorize-os categoricamente, referenciando prazos, cláusulas e limites.\n"
    "   - Diferencie explicitamente o que está disposto no documento carregado daquilo que é norma geral de mercado.\n"
    "2. Rigor Técnico e Terminologia:\n"
    "   - Utilize com precisão os conceitos de LMI, LMG, POS, Franquias Simples e Dedutíveis, Sub-rogação, Cosseguro e Resseguro.\n"
    "   - Mencione os artigos ou circulares pertinentes quando aplicável (ex.: Circular SUSEP 621, Decreto-Lei 73/66).\n"
    "3. Estilo e Tom:\n"
    "   - Postura executiva, objetiva, imparcial e formal em Português do Brasil.\n"
    "   - Organize explicações complexas em seções bem delimitadas e listas analíticas."
)

# ---------------------------------------------------------
# Barra Lateral Institucional
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ Painel de Controle")
    st.caption("Sistema de Inteligência Normativa & Sinistros")
    st.markdown("---")

    # Módulo RAG
    st.markdown("#### 📂 Repositório Documental (RAG)")
    uploaded_files = st.file_uploader(
        "Selecione Manuais, Apólices ou Circulares:",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        help="Permite a verificação semântica cruzada com a documentação fornecida.",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        index_clicked = st.button("📥 Indexar", use_container_width=True, type="primary")
    with col_btn2:
        clear_doc_clicked = st.button("🗑️ Descartar", use_container_width=True)

    if index_clicked:
        if uploaded_files:
            with st.spinner("Processando e vetorizando documentos..."):
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
                    st.session_state["doc_name"] = ", ".join([f.name for f in uploaded_files])
                    st.success(f"{len(chunks)} fragmentos normativos indexados.")
                else:
                    st.warning("Não foi possível extrair conteúdo textual dos arquivos.")
        else:
            st.info("Nenhum arquivo anexado para indexação.")

    if clear_doc_clicked and st.session_state.get("doc_loaded"):
        st.session_state.pop("rag_chunks", None)
        st.session_state.pop("rag_embeddings", None)
        st.session_state["doc_loaded"] = False
        st.rerun()

    if st.session_state.get("doc_loaded"):
        st.markdown(
            f"<div class='badge-rag-active'>✓ Base ativa: {st.session_state.get('doc_name', 'Documento')}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Consultas Padronizadas
    st.markdown("#### 📋 Consultas Regulatórias Rápidas")
    consultas = [
        "Qual o prazo regulamentar para liquidação de sinistros pela SUSEP?",
        "Qual o tratamento da SUSEP para suspensão de prazo de sinistro?",
        "Diferença entre LMI (Limite Máximo de Indenização) e LMG na apólice",
        "Como funciona a Cláusula de Sub-rogação de Direitos?",
    ]
    for consulta in consultas:
        if st.button(consulta, use_container_width=True):
            st.session_state["pending_prompt"] = consulta

    st.markdown("---")

    # Configurações do Foundry / Azure (em Expander para manter a sobriedade)
    with st.expander("⚙️ Parâmetros do Microsoft Foundry", expanded=False):
        api_endpoint = st.text_input("Endpoint do Serviço:", value=DEFAULT_AZURE_ENDPOINT)
        api_key = st.text_input("Chave Primária (API Key):", value=DEFAULT_AZURE_API_KEY, type="password")
        model_deployment = st.text_input("Modelo Implantado:", value=DEFAULT_AZURE_MODEL)

    if st.button("🔄 Reiniciar Sessão de Análise", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Cabeçalho Principal (Estilo Governamental / Institucional)
# ---------------------------------------------------------
st.markdown(
    """
    <div class="gov-header">
        <div>
            <div class="gov-title">PORTAL DE INTELIGÊNCIA TÉCNICA E REGULATÓRIA EM SEGUROS</div>
            <div class="gov-subtitle">Supervisão Técnica de Ramos Corporativos, Apólices e Procedimentos de Regulação</div>
        </div>
        <div class="gov-badge">Conformidade SNSP / SUSEP</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Card informativo institucional
st.markdown(
    """
    <div class="corp-card">
        <b>Ambiente de Consulta Especializada:</b> Utilize este módulo para analisar condições contratuais, 
        prazos de liquidação de sinistros, cálculos de franquias/POS e enquadramento regulatório.
        Para verificação pontual de apólices ou minutas, indexe o documento no painel lateral.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Histórico de Sessão
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bem-vindo ao canal corporativo de suporte técnico e regulatório de seguros. "
                "Estou pronto para analisar termos de apólices, sinistros e normativas do setor. "
                "Caso deseje uma análise personalizada, indexe o documento correspondente na barra lateral."
            ),
        }
    ]

for msg in st.session_state.messages:
    avatar = "⚖️" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

prompt_input = st.chat_input("Digite a consulta técnica, cláusula de apólice ou questão regulatória...")
if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    active_prompt = st.session_state.pop("pending_prompt")
else:
    active_prompt = prompt_input

# ---------------------------------------------------------
# Processamento e Streaming com AzureOpenAI
# ---------------------------------------------------------
if active_prompt:
    if not api_key or not api_endpoint:
        st.warning("⚠️ Parâmetros de autenticação do Microsoft Foundry incompletos. Verifique a barra lateral.")
    else:
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_prompt)

        # Busca no acervo indexado
        relevant_chunks = retrieve_relevant_chunks(active_prompt, top_k=3)
        context_block = ""
        if relevant_chunks:
            context_block = "\n\n--- [TRECHOS EXTRAÍDOS DOS DOCUMENTOS REGULATÓRIOS/APÓLICES] ---\n"
            for i, chunk in enumerate(relevant_chunks, start=1):
                context_block += f"[Cláusula/Trecho {i}]:\n{chunk}\n\n"
            context_block += "--- [FIM DO CONTEXTO DOCUMENTAL] ---\n"

        messages_payload = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        for m in st.session_state.messages[:-1]:
            messages_payload.append({"role": m["role"], "content": m["content"]})

        final_prompt = active_prompt
        if context_block:
            final_prompt = (
                f"{active_prompt}\n\n"
                f"Considere as disposições dos documentos indexados a seguir para fundamentar a análise:\n"
                f"{context_block}"
            )

        messages_payload.append({"role": "user", "content": final_prompt})

        with st.chat_message("assistant", avatar="⚖️"):
            try:
                clean_endpoint = api_endpoint.strip().rstrip('"').rstrip("/")

                client = AzureOpenAI(
                    azure_endpoint=clean_endpoint,
                    api_key=api_key.strip(),
                    api_version=AZURE_API_VERSION,
                )

                response_stream = client.chat.completions.create(
                    model=model_deployment.strip(),
                    messages=messages_payload,
                    temperature=0.1,  # Temperatura reduzida para máxima consistência técnica e regulatória
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
                st.error(f"Erro na comunicação com o Microsoft Foundry: {str(e)}")
