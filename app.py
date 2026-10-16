"""
Consultor Virtual Especialista em Seguros
Licença MIT - Código pronto para deploy no GitHub / Streamlit Community Cloud
"""

import os
import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Configuração da Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Consultor de Seguros AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Engenharia de Prompt (System Instruction Formatada de Forma Segura)
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "Você é o 'SegurAI', um consultor técnico e subscritor sênior especialista no mercado segurador brasileiro e internacional.\n"
    "Sua missão é auxiliar corretores, analistas corporativos, gestores de risco e segurados em dúvidas técnicas, operacionais e conceituais de seguros.\n\n"
    "Diretrizes de Atuação:\n"
    "1. Conhecimento Técnico Aprofundado:\n"
    "   - Domine ramos corporativos e elementares: D&O (Directors and Officers), E&O, Riscos de Engenharia, Riscos Operacionais, Property / Patrimonial, Responsabilidade Civil Geral (RC Geral), Seguro Garantia, Frotas, Vida em Grupo, Transporte e Resseguro.\n"
    "   - Conheça os termos de mercado: Franquia simples e dedutível, Participação Obrigatória do Segurado (POS), Limite Máximo de Garantia (LMG), Limite Máximo de Indenização (LMI), Cosseguro, Resseguro (Tratado e Facultativo), Sub-rogação, Regulação de Sinistros e Salvados.\n"
    "   - Faça referências adequadas ao marco regulatório brasileiro (Normas, Resoluções e Circulares SUSEP, além do Código Civil Brasileiro nos capítulos de Seguros).\n\n"
    "2. Tom e Estilo:\n"
    "   - Profissional, analítico, claro, didático e empático.\n"
    "   - Estruture respostas complexas em tópicos, destacando riscos cobertos, exclusões típicas e boas práticas na gestão de apólices e sinistros.\n"
    "   - Sempre forneça explicações fundamentadas tecnicamente, pontuando quando uma questão depende das Condições Gerais específicas da apólice contratada.\n"
    "   - Idioma principal: Português (Brasil)."
)

# ---------------------------------------------------------
# Obtenção da API Key (st.secrets ou input lateral)
# ---------------------------------------------------------
def get_api_key() -> str:
    """Busca a chave nos secrets ou variáveis de ambiente."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    return api_key


api_key_default = get_api_key()

# ---------------------------------------------------------
# Barra Lateral (Sidebar)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🛡️ SegurAI Analytics")
    st.caption("Assistente Técnico de Riscos & Seguros")
    st.markdown("---")

    # Gestão da Chave de API
    if not api_key_default:
        st.subheader("🔑 Autenticação")
        user_key = st.text_input(
            "Gemini API Key:",
            type="password",
            help="Obtenha sua chave gratuita no Google AI Studio (aistudio.google.com).",
        )
        api_key_to_use = user_key.strip()
    else:
        api_key_to_use = api_key_default
        st.success("API Key carregada com sucesso!")

    st.markdown("---")

    # Tópicos Rápidos / Exemplos de Prompts
    st.subheader("💡 Sugestões Rápidas")
    sugestoes = [
        "Diferença entre D&O e E&O na prática",
        "Como funciona a POS em Riscos de Engenharia?",
        "Passo a passo da regulação de sinistro de Frota",
        "O que cobre a cláusula de Responsabilidade Civil Cruzada?",
        "Qual o papel do cosseguro em grandes riscos patrimoniais?",
    ]

    for sugestao in sugestoes:
        if st.button(sugestao, use_container_width=True):
            st.session_state["pending_prompt"] = sugestao

    st.markdown("---")

    # Botão de Reset
    if st.button("🧹 Limpar Conversa", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Desenvolvido com Streamlit & Google Gemini 2.5 Flash")

# ---------------------------------------------------------
# Gerenciamento de Histórico (Session State)
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Olá! Sou o seu Consultor Virtual de Seguros. Posso ajudar na interpretação "
                "de coberturas, gestão de apólices corporativas, regulação de sinistros, "
                "ramos elementares (D&O, Riscos de Engenharia, RC Geral, Frotas, etc.) e normas do setor. "
                "Como posso apoiar sua operação hoje?"
            ),
        }
    ]

# ---------------------------------------------------------
# Área Central de Chat
# ---------------------------------------------------------
st.title("Consultor Virtual Especialista em Seguros")
st.markdown(
    "Tire dúvidas técnicas sobre coberturas, termos de mercado (POS, LMG/LMI, franquias), "
    "procedimentos de regulação e análise de riscos corporativos."
)

# Renderiza histórico
for msg in st.session_state.messages:
    avatar = "🛡️" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Tratamento de prompt pendente vindo dos botões da sidebar
prompt_input = st.chat_input("Digite sua dúvida sobre apólices, sinistros ou coberturas...")
if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    active_prompt = st.session_state.pop("pending_prompt")
else:
    active_prompt = prompt_input

# ---------------------------------------------------------
# Execução da Conversação com Streaming
# ---------------------------------------------------------
if active_prompt:
    if not api_key_to_use:
        st.error("⚠️ Insira uma chave de API válida na barra lateral ou nos secrets para continuar.")
    else:
        # 1. Registra mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_prompt)

        # 2. Prepara histórico para envio ao Gemini SDK
        contents_payload = []
        for m in st.session_state.messages:
            role_tag = "user" if m["role"] == "user" else "model"
            contents_payload.append(
                types.Content(
                    role=role_tag,
                    parts=[types.Part.from_text(text=m["content"])],
                )
            )

        # 3. Execução e Streaming de Resposta
        with st.chat_message("assistant", avatar="🛡️"):
            try:
                client = genai.Client(api_key=api_key_to_use)

                response_stream = client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=contents_payload,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,
                    ),
                )

                def stream_generator():
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text

                # Exibe a resposta com streaming nativo do Streamlit
                full_response = st.write_stream(stream_generator())

                # Salva a resposta no histórico da sessão
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Erro ao consultar a API: {str(e)}")
