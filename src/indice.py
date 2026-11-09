"""Criação/carregamento de índice FAISS com embeddings locais."""

from __future__ import annotations
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

DEFAULT_EMBEDDING_MODEL = "bge-m3"


def criar_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> OllamaEmbeddings:
    load_dotenv()

    return OllamaEmbeddings(
        model=model_name,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def criar_indice(
    chunks: list[Document],
    diretorio: str | Path = "data/index/faiss",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> FAISS:
    """Constrói e guarda o índice local. Não guardar PDFs nem pesos de modelos no Git."""
    if not chunks:
        raise ValueError("Não há chunks para indexar. Verifica a extração dos PDFs.")
    embeddings = criar_embeddings(model_name)
    indice = FAISS.from_documents(chunks, embeddings)
    indice.save_local(str(diretorio))
    return indice


def carregar_indice(
    diretorio: str | Path = "data/index/faiss",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> FAISS:
    """Carrega apenas índices locais criados/confiados pelo próprio grupo.

    FAISS serializa metadados auxiliares em pickle; nunca ativar a desserialização
    perigosa para artefactos descarregados de origem desconhecida.
    """
    caminho = Path(diretorio)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Índice não encontrado: {caminho}. Executa criar_indice primeiro."
        )
    return FAISS.load_local(
        str(caminho),
        criar_embeddings(model_name),
        allow_dangerous_deserialization=True,
    )


def criar_retriever(
    indice: FAISS,
    k: int = 5,
) -> VectorStoreRetriever:
    """Cria o recuperador. k=5 alinha com a métrica proposta Recall@5; validar no protocolo."""
    if k < 1:
        raise ValueError("k tem de ser pelo menos 1")
    return indice.as_retriever(search_kwargs={"k": k})
