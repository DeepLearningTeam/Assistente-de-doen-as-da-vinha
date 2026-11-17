import sys
import uuid
import re
import time
from contextlib import closing

from pathlib import Path
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector, Modifier, PayloadSchemaType

from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


from .indice import (NOME_VETOR_DENSO, NOME_VETOR_ESPARSO, DIM_VETOR,
                     obter_modelo_bge, obter_bm25, validar_colecao)
from .apoio import RAIZ, ErroProjeto, gravar_json, sha256
from .config import carregar_config
from .corpus import ler_corpus as corpus_validado, selecionadas, PASTA_PDF, verificar

# ── Configuração ───────────────────────────────────────────────────────────
PASTA_PDFS = PASTA_PDF
MIN_CHARS_CHUNK = 50       # descarta chunks quase vazios (rodapés, títulos soltos)
TAMANHO_LOTE_UPSERT = 128

def ler_corpus():
    # Reutiliza o corpus validado da aplicação e a proteção dos documentos reservados.
    return list(selecionadas(corpus_validado()).values())


def obter_pdf(entrada):
    destino = PASTA_PDFS / f"{entrada['doc_id']}.pdf"
    if not destino.is_file() or sha256(destino) != entrada["sha256"]:
        raise ErroProjeto(f"{entrada['doc_id']}: importa primeiro o ZIP com src.corpus importar.")
    return destino


def _norm_linha(linha: str) -> str:
    """Normaliza uma linha para comparação (deteção de cabeçalhos/rodapés repetidos)."""
    linha = re.sub(r"</?u>", "", linha)
    linha = re.sub(r"[*_#]", "", linha)
    return re.sub(r"\s+", " ", linha).strip().lower()

def detetar_linhas_repetidas(paginas_md, minimo_paginas=6, fracao=0.4):
    """Linhas curtas presentes em muitas páginas = cabeçalhos/rodapés correntes."""
    if len(paginas_md) < minimo_paginas:
        return set()
    contagem = Counter()
    for p in paginas_md:
        vistas = {_norm_linha(l) for l in p["text"].splitlines()}
        contagem.update(l for l in vistas if 0 < len(l) < 120)
    return {l for l, n in contagem.items() if n >= fracao * len(paginas_md)}

def limpar_pagina(texto: str, repetidas: set) -> str:
    """Remove ruído de formatação, marcadores de OCR, e cabeçalhos/rodapés repetidos."""

    # 1. Marcadores de texto extraído de imagens (OCR) — mantém o conteúdo, tira os marcadores
    texto = re.sub(r"<!--\s*(Start|End) of picture text\s*-->", "", texto)
    texto = texto.replace("<br>", "\n")
    texto = re.sub(r"</?mark>", "", texto)

    # 2. Markdown/HTML residual do pymupdf4llm
    texto = re.sub(r"</?u>", "", texto)                       # <u>...</u> → remove as tags
    texto = texto.replace("**", "")                           # negrito
    texto = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", texto)  # itálico _texto_ → texto

    # 3. Linha a linha: remove cabeçalhos/rodapés recorrentes e números de página isolados
    linhas_limpas = []
    for linha in texto.splitlines():
        n = _norm_linha(linha)
        if n in repetidas:
            continue
        if re.fullmatch(r"\d{1,4}", n):          # linha que é só um número (nº de página)
            continue
        linhas_limpas.append(linha)

    return "\n".join(linhas_limpas)

# ── Leitura ───────────────────────────────────────────────────────────────

def processar_corpus(entradas: list[dict], prefixar_titulo: bool = False, config=None, ocr="auto") -> list[dict]:
    """Converte os documentos do corpus em chunks, com metadados do corpus no payload."""
    import pymupdf4llm
    config = config or carregar_config()
    chunks = []
    opcoes_ocr = {}
    if ocr == "off":
        opcoes_ocr["use_ocr"] = False
    elif ocr == "por":
        from .corpus import escolher_ocr
        escolher_ocr("por")
        from pymupdf4llm.ocr import tesseract_api
        if tesseract_api.TESSDATA is None:
            raise ErroProjeto("Define TESSDATA_PREFIX para a pasta que contém por.traineddata.")
        opcoes_ocr = {"use_ocr": True, "ocr_language": "por", "ocr_function": tesseract_api.exec_ocr}

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Secao"), ("##", "Subsecao"), ("###", "Topico")]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_carateres, chunk_overlap=config.sobreposicao
    )

    for entrada in entradas:
        doc_id = entrada["doc_id"]
        caminho = obter_pdf(entrada)
        if caminho is None:
            continue

        repetidas = set()
        try:
            paginas_md = pymupdf4llm.to_markdown(
                str(caminho), page_chunks=True, write_images=False, **opcoes_ocr
            )
            repetidas = detetar_linhas_repetidas(paginas_md)
            print(f"{doc_id}: {len(repetidas)} linhas repetidas detetadas, "
                f"ex.: {sorted(repetidas)[:3]}")
        except Exception as e:
            if ocr == "por":
                raise ErroProjeto(f"Falhou o OCR português de {doc_id}; não foi aplicado fallback silencioso.") from None
            if "NoneType' object is not iterable" in str(e):
                print(f"AVISO: OCR falhou em {doc_id} (imagem sem texto detetável); "
                      f"a repetir sem OCR.")
                try:
                    paginas_md = pymupdf4llm.to_markdown(
                        str(caminho), page_chunks=True, write_images=False, use_ocr=False
                    )
                except Exception as e2:
                    print(f"Falha definitiva ao ler {doc_id} mesmo sem OCR: {e2}")
                    continue   # salta este documento, mas continua o corpus
            else:
                print(f"Falha ao ler {doc_id}: {e}")
                continue

        repetidas = detetar_linhas_repetidas(paginas_md)

        # Verificação da numeração de páginas contra o corpus
        if entrada["paginas"].strip().isdigit() and len(paginas_md) != int(entrada["paginas"]):
            print(f"AVISO: {doc_id} tem {len(paginas_md)} páginas no PDF, "
                  f"o corpus indica {entrada['paginas']}.")

        meta_doc = {
            "doc_id": doc_id,
            "nome_documento": f"{doc_id}.pdf", 
            "titulo": entrada["titulo"],
            "entidade": entrada["entidade"],
            "data": entrada["data"],
            "url": entrada["url"],
            "temas": [t.strip() for t in entrada["temas"].split(";") if t.strip()],
            "estado_fonte": entrada["indexar"].strip().lower(),
        }

        # enumerate(start=1): a página é a posição na lista (1-based), sem
        # depender da chave de metadata, que varia entre versões do pymupdf4llm.
        for num_pagina, pagina in enumerate(
            tqdm(paginas_md, desc=f"A processar {doc_id}"), start=1
        ):
            texto_limpo = limpar_pagina(pagina["text"], repetidas)
            chunks_logicos = markdown_splitter.split_text(texto_limpo)
            chunks_finais = text_splitter.split_documents(chunks_logicos)

            for frag in chunks_finais:
                conteudo = frag.page_content.strip()
                if len(conteudo) < MIN_CHARS_CHUNK:
                    continue

                cabecalhos = " > ".join(frag.metadata.values())
                if prefixar_titulo:
                    cabecalhos = f"{entrada['titulo']} > {cabecalhos}" if cabecalhos else entrada["titulo"]
                texto = f"[{cabecalhos}]\n{conteudo}" if cabecalhos else conteudo

                chunks.append({
                    "texto": texto,
                    "pagina": num_pagina,
                    "cabecalhos": cabecalhos,
                    **meta_doc,
                })

    return chunks

# ── Indexação ──────────────────────────────────────────────────────────────
def construir_base_qdrant(lista_chunks: list[dict], client: QdrantClient, config=None) -> None:
    """Vetoriza os chunks e faz upsert (idempotente, em lotes) no Qdrant."""
    from qdrant_client.local.qdrant_local import QdrantLocal
    config = config or carregar_config()
    # Este fluxo nunca escreve no serviço partilhado, mesmo se chamado por Python.
    if not isinstance(client._client, QdrantLocal):
        raise ErroProjeto("A construção está limitada ao Qdrant local.")
    if not lista_chunks:
        raise ValueError("Não há texto para vetorizar: a lista de chunks está vazia.")

    invalidos = [c for c in lista_chunks if not c.get("texto")]
    if invalidos:
        print(f"AVISO: {len(invalidos)} chunks sem texto válido, a ignorar. "
              f"Exemplos: {[(c.get('doc_id'), c.get('pagina')) for c in invalidos[:5]]}")
        lista_chunks = [c for c in lista_chunks if c.get("texto")]

    if not client.collection_exists(config.qdrant_colecao):
        client.create_collection(
            collection_name=config.qdrant_colecao,
            sparse_vectors_config={NOME_VETOR_ESPARSO: SparseVectorParams(modifier=Modifier.IDF)},
            vectors_config={NOME_VETOR_DENSO: VectorParams(size=DIM_VETOR, distance=Distance.COSINE)},
        )
        for campo in ("nome_documento", "doc_id", "estado_fonte"):
            client.create_payload_index(config.qdrant_colecao, field_name=campo,
                                        field_schema=PayloadSchemaType.KEYWORD)

    validar_colecao(client, config)
    textos = [c["texto"] for c in lista_chunks]
    
    # Vetores esparsos
    t0 = time.perf_counter()
    vetores_esparsos = list(obter_bm25().passage_embed(textos))
    print(f"BM25: {time.perf_counter() - t0:.0f}s")
    
    # Vetores densos
    t0 = time.perf_counter()
    vetores_densos = obter_modelo_bge(config.embeddings_modelo).encode(
        textos,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    print(f"Embeddings densos: {time.perf_counter() - t0:.0f}s")

    pontos = [
        PointStruct(
            id=str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{c['nome_documento']}|{c['pagina']}|{c['texto']}",
            )),
            vector={
                NOME_VETOR_DENSO: vetores_densos[i].tolist(),
                NOME_VETOR_ESPARSO: SparseVector(
                    indices=vetores_esparsos[i].indices.tolist(),
                    values=vetores_esparsos[i].values.tolist(),
                    ),
                },
            payload=c,
        )
        for i, c in enumerate(lista_chunks)
    ]
    t0 = time.perf_counter()
    for i in range(0, len(pontos), TAMANHO_LOTE_UPSERT):
        client.upsert(
            collection_name=config.qdrant_colecao,
            points=pontos[i:i + TAMANHO_LOTE_UPSERT],
        )
    print(f"Upsert: {time.perf_counter() - t0:.0f}s")
    print(f"Upsert concluído: {len(pontos)} pontos na coleção '{config.qdrant_colecao}'.")

# ── Execução direta ────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Construir uma coleção local nova; nunca altera a coleção do grupo.")
    parser.add_argument("--local", type=Path, default=RAIZ / "data/qdrant_local")
    parser.add_argument("--colecao", default="videira_teste")
    parser.add_argument("--ocr", choices=["auto", "por", "off"], default="auto")
    args = parser.parse_args()
    config = carregar_config().model_copy(update={"qdrant_colecao": args.colecao})
    try:
        entradas = ler_corpus()
        verificar({e["doc_id"]: e for e in entradas})
        # Fecha os ficheiros locais mesmo se ocorrer uma falha de extração.
        with closing(QdrantClient(path=str(args.local))) as client:
            if client.collection_exists(args.colecao):
                raise ErroProjeto("A coleção já existe. Usa outro --local ou --colecao; não há apagamento automático.")
            chunks = processar_corpus(entradas, config=config, ocr=args.ocr)
            if {c["doc_id"] for c in chunks} != {e["doc_id"] for e in entradas}:
                raise ErroProjeto("Extração incompleta: nem todos os documentos produziram passagens.")
            construir_base_qdrant(chunks, client, config)
            gravar_json(args.local / f"{args.colecao}_ingestao.json", {
                "config": config.model_dump(), "chunks": len(chunks), "ocr": args.ocr,
                "corpus_sha256": sha256(RAIZ / "data/corpus.csv"),
                "pdfs": {e["doc_id"]: e["sha256"] for e in entradas},
            })
    except Exception as erro:
        sys.exit(str(erro) if isinstance(erro, ErroProjeto) else "Falhou a extração/indexação local. Confirma as dependências e os PDF.")


if __name__ == "__main__":
    main()
