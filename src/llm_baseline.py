"""Execução do LLM textual sem RAG (baseline do Tema F)."""
from __future__ import annotations
import os
import time
from typing import Any
from dotenv import load_dotenv
from src.prompts import BASELINE_SYSTEM_PROMPT


def criar_llm_local(model_name: str | None = None, temperature: float = 0.0):
    """Cria ChatOllama. Não liga ferramentas nem pesquisa documental."""
    load_dotenv()
    from langchain_ollama import ChatOllama
    nome = model_name or os.getenv('OLLAMA_MODEL', 'qwen3:1.7b')
    if not nome or nome.startswith(('PREENCHER', 'ESCOLHER')):
        raise RuntimeError('Define OLLAMA_MODEL no ficheiro .env.')
    return ChatOllama(
        model=nome,
        base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        temperature=temperature,
    )


def _texto_conteudo(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = []
        for item in content:
            if isinstance(item, dict):
                partes.append(str(item.get('text', item)))
            else:
                partes.append(str(item))
        return '\n'.join(partes)
    return str(content)


def responder_sem_rag(llm: Any, pergunta: str, temperature: float = 0.0) -> dict[str, Any]:
    """Executa uma pergunta sem retriever, sem CNN e sem qualquer ferramenta."""
    from langchain_core.messages import HumanMessage, SystemMessage
    inicio = time.perf_counter()
    resposta = llm.invoke([
        SystemMessage(content=BASELINE_SYSTEM_PROMPT),
        HumanMessage(content=pergunta),
    ])
    latencia = time.perf_counter() - inicio
    return {
        'resposta': _texto_conteudo(getattr(resposta, 'content', resposta)),
        'latencia_s': round(latencia, 4),
        'model_id': getattr(llm, 'model', os.getenv('OLLAMA_MODEL', 'desconhecido')),
        'temperature': float(temperature),
        'modo': 'sem_rag_sem_ferramentas',
    }
