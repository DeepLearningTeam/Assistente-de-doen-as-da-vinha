"""Agente LangChain com ferramentas CNN (F2) e RAG (F3)."""
from __future__ import annotations
import json
import os
from typing import Any, Callable
from PIL import Image
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from src.prompts import TOOLS_SYSTEM_PROMPT
from src.rag import formatar_contexto


def criar_llm_local(model_name: str | None = None, base_url: str | None = None):
    """Inicializa um LLM local no Ollama; o modelo tem de suportar tool calling."""
    load_dotenv()
    from langchain_ollama import ChatOllama
    nome = model_name or os.getenv('OLLAMA_MODEL', 'qwen3:1.7b')
    if not nome or nome.startswith(('PREENCHER', 'ESCOLHER')):
        raise RuntimeError('Define OLLAMA_MODEL no ficheiro .env.')
    return ChatOllama(
        model=nome,
        base_url=base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        temperature=0.0,
    )


def criar_agente_llm_tools(
    llm: Any,
    retriever: Any,
    imagem: Image.Image | None = None,
    classificador: Callable[[Image.Image], dict[str, Any]] | None = None,
):
    """Cria um agente por pedido. A ferramenta CNN só existe se houver imagem.

    Contrato esperado do classificador (F2):
      {'classe': 'esca', 'probabilidades': {'saudavel': 0.1, 'esca': 0.7, ...}}

    Contrato esperado do retriever (F3): retriever.invoke(texto) -> list[Document],
    com metadata: doc_id, titulo, entidade, data, pagina e url.
    """
    ferramentas = []

    @tool
    def pesquisar_documentacao(consulta: str) -> str:
        """Pesquisa passagens dos documentos técnicos indexados em português.

        Args:
            consulta: pergunta, doença ou tema a procurar no corpus do grupo.
        """
        documentos = list(retriever.invoke(consulta))
        if not documentos:
            return 'SEM EVIDÊNCIA: a pesquisa não devolveu passagens.'
        return formatar_contexto(documentos)

    ferramentas.append(pesquisar_documentacao)

    if imagem is not None:
        if classificador is None:
            raise ValueError('Foi fornecida uma imagem, mas a CNN do F2 não foi ligada.')

        @tool
        def classificar_folha() -> str:
            """Classifica a imagem anexada ao pedido através da CNN do projeto."""
            resultado = classificador(imagem)
            return json.dumps(resultado, ensure_ascii=False, default=str)

        ferramentas.append(classificar_folha)

    return create_agent(
        model=llm,
        tools=ferramentas,
        system_prompt=TOOLS_SYSTEM_PROMPT,
    )


def _conteudo_texto(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(str(x.get('text', x)) if isinstance(x, dict) else str(x) for x in content)
    return str(content)


def responder_com_agente(
    llm: Any,
    retriever: Any,
    pergunta: str,
    imagem: Image.Image | None = None,
    classificador: Callable[[Image.Image], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    agente = criar_agente_llm_tools(llm, retriever, imagem, classificador)
    resultado = agente.invoke({'messages': [{'role': 'user', 'content': pergunta}]})
    mensagens = resultado.get('messages', [])
    final = ''
    for msg in reversed(mensagens):
        if getattr(msg, 'type', '') == 'ai' and getattr(msg, 'content', None):
            final = _conteudo_texto(msg.content)
            break
    chamadas = []
    for msg in mensagens:
        for chamada in getattr(msg, 'tool_calls', []) or []:
            chamadas.append(chamada.get('name', 'ferramenta_desconhecida'))
        if getattr(msg, 'type', '') == 'tool':
            chamadas.append(getattr(msg, 'name', 'ferramenta'))
    return {'resposta': final, 'ferramentas_invocadas': chamadas, 'mensagens': mensagens}
