import os
import sys
import uuid
import re
import time

from pathlib import Path
from collections import Counter
import hashlib
import csv
import requests


from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector, Modifier, Prefetch, FusionQuery, Fusion, PayloadSchemaType

import pymupdf4llm
from pymupdf4llm.ocr import rapidocr_onnx_backend
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

# ── Configuração ───────────────────────────────────────────────────────────
PASTA_PDFS = Path(__file__).resolve().parent / "pdfs_cache"
INDEXAR_VALORES = {"sim", "condicional"} 

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Faltam QDRANT_URL e/ou QDRANT_API_KEY no ficheiro .env")

NOME_COLECAO = "videira_fitossanidade"
MODELO_EMBEDDINGS = "BAAI/bge-m3"
MODELO_RERANKER = "BAAI/bge-reranker-v2-m3"
DIM_VETOR = 1024
MODELO_LLM = "phi3"

CHUNK_SIZE = 1200          # caracteres (~300 tokens)
CHUNK_OVERLAP = 200        # caracteres (~50 tokens)
MIN_CHARS_CHUNK = 50       # descarta chunks quase vazios (rodapés, títulos soltos)
TOP_K = 5
K_CANDIDATOS=30
TAMANHO_LOTE_UPSERT = 128
LIMIAR_RERANKER = 0.60    # documentar a escolha e a limitação da amostra

_reranker = None

LLM_OPCOES = {"num_ctx": 8192, "temperature": 0}
MSG_SEM_INFO = "Não há informação disponível na base de dados para este problema."

NOME_VETOR_DENSO = "vetor_denso"
NOME_VETOR_ESPARSO = "bm25"

_bm25 = None


# ── Modelo de embeddings ─────────────────────────
_modelo_bge = None

def obter_modelo_bge() -> SentenceTransformer:
    """Carrega o bge-m3 só na primeira utilização (evita ~2 GB em cada import)."""
    global _modelo_bge
    if _modelo_bge is None:
        _modelo_bge = SentenceTransformer(MODELO_EMBEDDINGS)
    return _modelo_bge

def obter_bm25() -> SparseTextEmbedding:
    global _bm25
    if _bm25 is None:
        _bm25 = SparseTextEmbedding("Qdrant/bm25", language="portuguese")
    return _bm25

def obter_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(MODELO_RERANKER, max_length=512)
    return _reranker

def _full_ocr_seguro(img):
    engine = rapidocr_onnx_backend.init_engine()
    results, times = engine(img)
    if not results:
        return []
    return [(box, text, float(score)) for box, text, score in results]

rapidocr_onnx_backend.full_ocr = _full_ocr_seguro

# ── Ingestão ───────────────────────────────────────────────────────────────
def ler_corpus(caminho_csv: str, valores_indexar=INDEXAR_VALORES) -> list[str]:
    """Lê o corpus.csv e devolve só as linhas cuja coluna 'indexar' está em valores_indexar."""
    with open(caminho_csv, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f))

    selecionadas = []
    for l in linhas:
        if l["indexar"].strip().lower() not in valores_indexar:
            continue
        if not l["url"].strip():
            print(f"AVISO: {l['doc_id']} sem URL, ignorado.")
            continue
        selecionadas.append(l)
    return selecionadas

def _sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()

def obter_pdf(entrada: dict) -> Path | None:
    """Garante o PDF em pdfs_cache/<doc_id>.pdf (descarrega só se ainda não existir)."""
    PASTA_PDFS.mkdir(exist_ok=True)
    destino = PASTA_PDFS / f"{entrada['doc_id']}.pdf"

    if not destino.exists():
        try:
            # User-Agent de browser: alguns servidores recusam o do 'requests'
            r = requests.get(entrada["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"Falha a descarregar {entrada['doc_id']}: {e}")
            return None
        if not r.content.startswith(b"%PDF"):     # páginas de erro/login vêm com status 200
            print(f"{entrada['doc_id']}: a resposta não é um PDF (URL mudou?).")
            return None
        destino.write_bytes(r.content)

    esperado = entrada["sha256"].strip().lower()
    if esperado and _sha256(destino) != esperado:
        print(f"AVISO: sha256 de {entrada['doc_id']} não bate com o corpus "
              f"(documento atualizado, ou hash por revalidar).")
    return destino

# ── Limpeza ───────────────────────────────────────────────────────────────
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

def processar_corpus(entradas: list[dict], prefixar_titulo: bool = False) -> list[dict]:
    """Converte os documentos do corpus em chunks, com metadados do corpus no payload."""
    chunks = []

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Secao"), ("##", "Subsecao"), ("###", "Topico")]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    for entrada in entradas:
        doc_id = entrada["doc_id"]
        caminho = obter_pdf(entrada)
        if caminho is None:
            continue

        try:
            paginas_md = pymupdf4llm.to_markdown(
                str(caminho), page_chunks=True, write_images=False
            )
            repetidas = detetar_linhas_repetidas(paginas_md)
            print(f"{doc_id}: {len(repetidas)} linhas repetidas detetadas, "
                f"ex.: {sorted(repetidas)[:3]}")
        except Exception as e:
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
def construir_base_qdrant(lista_chunks: list[dict], client: QdrantClient) -> None:
    """Vetoriza os chunks e faz upsert (idempotente, em lotes) no Qdrant."""
    if not lista_chunks:
        raise ValueError("Não há texto para vetorizar: a lista de chunks está vazia.")

    invalidos = [c for c in lista_chunks if not c.get("texto")]
    if invalidos:
        print(f"AVISO: {len(invalidos)} chunks sem texto válido, a ignorar. "
              f"Exemplos: {[(c.get('doc_id'), c.get('pagina')) for c in invalidos[:5]]}")
        lista_chunks = [c for c in lista_chunks if c.get("texto")]

    if not client.collection_exists(NOME_COLECAO):
        client.create_collection(
            collection_name=NOME_COLECAO,
            sparse_vectors_config={NOME_VETOR_ESPARSO: SparseVectorParams(modifier=Modifier.IDF)},
            vectors_config={NOME_VETOR_DENSO: VectorParams(size=DIM_VETOR, distance=Distance.COSINE)},
        )
        for campo in ("nome_documento", "doc_id", "estado_fonte"):
            client.create_payload_index(NOME_COLECAO, field_name=campo,
                                        field_schema=PayloadSchemaType.KEYWORD)

    textos = [c["texto"] for c in lista_chunks]
    
    # Vetores esparsos
    t0 = time.perf_counter()
    vetores_esparsos = list(obter_bm25().passage_embed(textos))
    print(f"BM25: {time.perf_counter() - t0:.0f}s")
    
    # Vetores densos
    t0 = time.perf_counter()
    vetores_densos = obter_modelo_bge().encode(
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
            collection_name=NOME_COLECAO,
            points=pontos[i:i + TAMANHO_LOTE_UPSERT],
        )
    print(f"Upsert: {time.perf_counter() - t0:.0f}s")
    print(f"Upsert concluído: {len(pontos)} pontos na coleção '{NOME_COLECAO}'.")

# ── Recuperação ────────────────────────────────────────────────────────────
def recuperar_candidatos(pergunta: str, client: QdrantClient,
                         modo: str = "hibrida", k: int = K_CANDIDATOS) -> list[dict]:
    """Fase 1 (Qdrant): devolve k candidatos, sem reranking."""
    denso = obter_modelo_bge().encode([pergunta], normalize_embeddings=True)[0].tolist()

    if modo == "denso":
        resposta = client.query_points(
            collection_name=NOME_COLECAO, query=denso,
            using=NOME_VETOR_DENSO, limit=k,
        )
    else:
        sp = next(iter(obter_bm25().query_embed(pergunta)))
        resposta = client.query_points(
            collection_name=NOME_COLECAO,
            prefetch=[
                Prefetch(query=denso, using=NOME_VETOR_DENSO, limit=k),
                Prefetch(query=SparseVector(indices=sp.indices.tolist(),
                                            values=sp.values.tolist()),
                         using=NOME_VETOR_ESPARSO, limit=k),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
        )
    return [{**h.payload, "id": h.id, "score_pesquisa": h.score} for h in resposta.points]

def reranquear(pergunta: str, candidatos: list[dict]) -> list[dict]:
    """Fase 2 (Python): o cross-encoder lê (pergunta, chunk) em conjunto e reordena."""
    if not candidatos:
        return []
    scores = obter_reranker().predict(
        [(pergunta, c["texto"]) for c in candidatos], batch_size=16
    )
    for c, s in zip(candidatos, scores):
        c["score_reranker"] = float(s)
    return sorted(candidatos, key=lambda c: c["score_reranker"], reverse=True)

def pesquisar_query_qdrant(pergunta: str, client: QdrantClient,
                           modo: str = "hibrida", usar_reranker: bool = True,
                           k_candidatos: int = K_CANDIDATOS,
                           limiar: float | None = None) -> list[dict]:
    candidatos = recuperar_candidatos(pergunta, client, modo, k_candidatos)

    if usar_reranker:
        candidatos = reranquear(pergunta, candidatos)
        for c in candidatos:
            c["score"] = c["score_reranker"]
        corte = LIMIAR_RERANKER if limiar is None else limiar
        if corte is not None:                      # o corte só vale para o score do reranker
            candidatos = [c for c in candidatos if c["score"] >= corte]
    else:
        for c in candidatos:
            c["score"] = c["score_pesquisa"]

    return candidatos[:TOP_K]
 

# ── Execução direta ────────────────────────────────────────────────────────
if __name__ == "__main__":
    diretoria_script = Path(__file__).parent.parent / "data"
    caminho_corpus = os.path.join(diretoria_script, "corpus.csv")

    cliente = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if "--recriar" in sys.argv and cliente.collection_exists(NOME_COLECAO):
        cliente.delete_collection(NOME_COLECAO)
        print(f"Coleção '{NOME_COLECAO}' apagada.")
    
    if "--indexar" in sys.argv:
        t0 = time.perf_counter()
        entradas = ler_corpus(caminho_corpus)
        print(f"{len(entradas)} documentos: {[e['doc_id'] for e in entradas]}")
        chunks = processar_corpus(entradas)
        print(f"Corpus → chunks: {time.perf_counter() - t0:.0f}s "
              f"({len(chunks)} chunks)")
        construir_base_qdrant(chunks, cliente)