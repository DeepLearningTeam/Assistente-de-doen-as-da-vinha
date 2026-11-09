from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.corpus import extrair_paginas_pdf, dividir_em_chunks
from src.indice import criar_indice


def main():
    parser = argparse.ArgumentParser(
        description="Extrai PDFs e constrói o índice FAISS."
    )

    parser.add_argument(
        "--embedding",
        default="bge-m3",
        help="Modelo de embeddings instalado no Ollama.",
    )

    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--output", default="data/index/faiss")

    args = parser.parse_args()

    with open("data/corpus.csv", encoding="utf-8-sig", newline="") as f:
        corpus = {row["doc_id"]: row for row in csv.DictReader(f)}

    with open(
        "data/docs/mapa_ficheiros_locais.csv", encoding="utf-8-sig", newline=""
    ) as f:
        mapa = list(csv.DictReader(f))

    docs = []

    for item in mapa:
        doc_id = item["doc_id"]
        estado = item["indexacao"]

        # Política conservadora: só documentos aprovados são indexados.
        if estado != "sim":
            print(
                f"SKIP {doc_id}: estado '{estado}'. "
                "Apenas documentos com estado 'sim' são elegíveis."
            )
            continue

        if doc_id not in corpus:
            raise ValueError(f"Documento {doc_id} não encontrado em data/corpus.csv.")

        ficheiro = Path("data/docs") / item["nome_ficheiro_local"]

        if not ficheiro.exists():
            raise FileNotFoundError(f"Falta o PDF: {ficheiro}")

        meta = corpus[doc_id]

        metadados = {
            "doc_id": doc_id,
            "titulo": meta["titulo"],
            "entidade": meta["entidade"],
            "data": meta["data"],
            "url": meta["url"],
            "licenca": meta["licenca"],
        }

        paginas = extrair_paginas_pdf(ficheiro, metadados)

        chunks = dividir_em_chunks(paginas, args.chunk_size, args.chunk_overlap)

        print(f"{doc_id}: {len(paginas)} páginas com texto " f"-> {len(chunks)} chunks")

        docs.extend(chunks)

    if not docs:
        raise RuntimeError("Nenhum conteúdo elegível para indexar.")

    criar_indice(docs, args.output, args.embedding)

    print("\nÍndice FAISS criado com sucesso.")
    print(f"Diretório: {args.output}")
    print(f"Modelo de embeddings: {args.embedding}")
    print(f"Total de chunks: {len(docs)}")
    print("Documentos incluídos:")
    print(", ".join(sorted({doc.metadata["doc_id"] for doc in docs})))


if __name__ == "__main__":
    main()
