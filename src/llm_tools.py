"""Agente LangChain com ferramentas CNN e pesquisa documental.

O LLM precisa de suportar tool calling. A chamada da CNN só é disponibilizada quando a
pergunta atual inclui uma imagem, evitando que o agente peça uma imagem inexistente.
"""
from __future__ import annotations
import json
from typing import Any, Callable
from PIL import Image
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document

from src.rag import formatar_contexto

SYSTEM_PROMPT = """És um assistente técnico de apoio à vinha. Responde em português de Portugal.
Usa pesquisar_documentacao antes de fazer recomendações técnicas. Se houver uma imagem
anexada, usa classificar_folha para obter uma hipótese visual, mas não trates a classificação
como confirmação clínica. Baseia as recomendações nas passagens recuperadas; cita os IDs
[S1], [S2], etc. Não inventes fontes, produtos autorizados, doses ou datas. Se não houver
suporte documental suficiente, recusa a recomendação específica e explica que falta evidência.
"""


def criar_llm_local(model_name: str | None = None, base_url: str | None = None):
    """Inicializa um LLM servido pelo Ollama, sem API comercial.

    Requer: Ollama instalado/em execução e um modelo descarregado que suporte tool calling.
    O facto de um modelo abrir não significa que chame ferramentas corretamente; testar antes.
    """
    import os
    from langchain_ollama import ChatOllama

    nome = model_name or os.getenv("OLLAMA_MODEL", "")
    if not nome or nome.startswith("ESCOLHER") or nome.startswith("PREENCHER"):
        raise RuntimeError(
            "Define OLLAMA_MODEL no .env depois de escolher/testar um modelo local com tool calling."
        )
    url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=nome, base_url=url, temperature=0)


def criar_agente_llm_tools(
    llm: Any,
    retriever: Any,
    imagem: Image.Image | None = None,
    classificador: Callable[[Image.Image], dict[str, Any]] | None = None,
):
    """Cria um agente por pedido, com ferramentas disponíveis para esse pedido.

    `llm`: modelo LangChain que suporte chamadas de ferramentas.
    `retriever`: recuperador do F3/FAISS.
    `classificador`: função real ligada à CNN do F2.

    Nota: as imagens não são passadas como argumentos JSON do agente; a função mantém a
    imagem do pedido no seu âmbito. Não guardar a imagem em logs, salvo decisão explícita.
    """
    ferramentas = []

    @tool
    def pesquisar_documentacao(consulta: str) -> str:
        """Pesquisa passagens dos documentos técnicos de vinha relevantes para a consulta.

        Args:
            consulta: pergunta ou tema a procurar nos documentos indexados.
        """
        documentos = list(retriever.invoke(consulta))
        if not documentos:
            return "Não foram recuperadas passagens documentais para esta consulta."
        return formatar_contexto(documentos)

    ferramentas.append(pesquisar_documentacao)

    if imagem is not None:
        if classificador is None:
            raise ValueError("Foi fornecida uma imagem, mas não foi ligada a função CNN do F2.")

        @tool
        def classificar_folha() -> str:
            """Classifica a imagem de folha anexada ao pedido usando a CNN do projeto."""
            resultado = classificador(imagem)
            return json.dumps(resultado, ensure_ascii=False, default=str)

        ferramentas.append(classificar_folha)

    return create_agent(
        model=llm,
        tools=ferramentas,
        system_prompt=SYSTEM_PROMPT,
    )


def responder_com_agente(
    llm: Any,
    retriever: Any,
    pergunta: str,
    imagem: Image.Image | None = None,
    classificador: Callable[[Image.Image], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Executa o agente e devolve mensagens para inspeção/avaliação posterior."""
    agente = criar_agente_llm_tools(llm, retriever, imagem, classificador)
    resultado = agente.invoke({"messages": [{"role": "user", "content": pergunta}]})
    mensagens = resultado.get("messages", [])
    texto_final = ""
    for msg in reversed(mensagens):
        if getattr(msg, "type", "") == "ai" and getattr(msg, "content", None):
            texto_final = msg.content
            if isinstance(texto_final, list):
                texto_final = "\n".join(str(x) for x in texto_final)
            break
    return {"resposta": texto_final, "mensagens": mensagens}
