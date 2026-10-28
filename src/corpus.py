"""Extração de PDFs por página e chunking, preservando metadados para citações."""
from __future__ import annotations
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extrair_paginas_pdf(
    caminho_pdf: str | Path,
    metadados: dict[str, Any],
) -> list[Document]:
    """Extrai texto por página. `pagina` é 1-based para corresponder ao PDF aberto.

    Páginas sem texto (p.ex., slides digitalizados) são omitidas e reportadas via aviso;
    o OCR tem de ser decidido/implementado à parte, não é feito automaticamente aqui.
    `metadados` deve incluir doc_id, titulo, entidade, data, url e licenca verificados.
    """
    caminho = Path(caminho_pdf)
    documentos: list[Document] = []
    with fitz.open(caminho) as pdf:
        for indice, pagina_pdf in enumerate(pdf, start=1):
            texto = pagina_pdf.get_text("text").strip()
            if not texto:
                print(f"AVISO: página {indice} sem texto extraível em {caminho.name}; verificar OCR.")
                continue
            metadata = dict(metadados)
            metadata.update({"pagina": indice, "ficheiro_local": caminho.name})
            documentos.append(Document(page_content=texto, metadata=metadata))
    return documentos


def dividir_em_chunks(
    documentos: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Divide páginas em fragmentos, mantendo os metadados de documento/página.

    Os valores são um ponto de partida experimental, não valores exigidos pelo guia.
    Ajustar apenas com dados de desenvolvimento e registar a configuração usada.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size tem de ser positivo")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap tem de ser >=0 e menor que chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documentos)
