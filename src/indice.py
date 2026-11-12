"""Construção e pesquisa semântica do índice FAISS com BGE-M3 no Ollama."""
import argparse
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from .apoio import ErroProjeto, RAIZ, agora, escrever_atomico, sha256
from .config import EXPERIENCIAS, carregar_config
from .corpus import CSV_CORPUS, escolher_ocr, extrair_passagens, ler_corpus, selecionadas
from .modelos import digest_modelo, embeddings

FICHEIRO_INDICE = RAIZ / "data/indice.npz"
VERSAO = 1


def matriz_normalizada(vetores):
    matriz = np.ascontiguousarray(vetores, dtype=np.float32)
    if matriz.ndim != 2 or not len(matriz) or not np.isfinite(matriz).all():
        raise ErroProjeto("O Ollama devolveu vetores de embeddings inválidos.")
    if (np.linalg.norm(matriz, axis=1) == 0).any():
        raise ErroProjeto("O índice contém um vetor de embeddings vazio.")
    faiss.normalize_L2(matriz)
    return matriz


def construir(config, caminho=FICHEIRO_INDICE, opcao_ocr="auto", lote=24):
    digest = digest_modelo(config, config.embeddings_modelo)
    ocr = escolher_ocr(opcao_ocr)
    versao_ocr = None
    if ocr:
        proc = subprocess.run([ocr, "--version"], capture_output=True, text=True, check=True)
        versao_ocr = proc.stdout.splitlines()[0]
    print(f"Extração de PDF · OCR: {versao_ocr or 'desativado'}", flush=True)
    corpus_hash = sha256(CSV_CORPUS)
    passagens, documentos = extrair_passagens(config, ocr)
    if not passagens:
        raise ErroProjeto("Não foi possível extrair passagens dos PDF.")
    vetores = []
    for inicio in range(0, len(passagens), lote):
        textos = [p["texto"] for p in passagens[inicio:inicio + lote]]
        vetores.extend(embeddings(config, textos))
        print(f"Embeddings: {len(vetores)}/{len(passagens)}", flush=True)
    matriz = matriz_normalizada(vetores)
    if len(matriz) != len(passagens) or corpus_hash != sha256(CSV_CORPUS):
        raise ErroProjeto("O corpus mudou durante a indexação ou faltam embeddings.")
    if digest != digest_modelo(config, config.embeddings_modelo):
        raise ErroProjeto("O modelo de embeddings mudou durante a indexação.")
    metadados = {
        "versao": VERSAO, "criado_em": agora(), "corpus_sha256": corpus_hash,
        "embeddings": config.embeddings_modelo, "embeddings_digest": digest,
        "chunk_carateres": config.chunk_carateres, "sobreposicao": config.sobreposicao,
        "ocr": versao_ocr, "documentos": documentos, "passagens": passagens,
    }
    # Um único ficheiro, sem pickle. O índice FAISS é reconstruído ao abrir a aplicação.
    escrever_atomico(caminho, lambda f: np.savez_compressed(
        f, vetores=matriz, metadados=json.dumps(metadados, ensure_ascii=False),
    ))
    print(f"Índice criado em {caminho} ({len(documentos)} documentos, {len(matriz)} passagens).")


@lru_cache(maxsize=4)
def abrir_indice(caminho, modificacao_ns, tamanho):
    """A data e o tamanho fazem a aplicação recarregar um índice reconstruído."""
    with np.load(caminho, allow_pickle=False) as dados:
        metadados = json.loads(dados["metadados"].item())
        matriz = dados["vetores"]
    if metadados["versao"] != VERSAO or len(matriz) != len(metadados["passagens"]):
        raise ValueError("Versão ou dimensões do índice inválidas.")
    indice = faiss.IndexFlatIP(matriz.shape[1])
    indice.add(matriz)
    return indice, metadados


def abrir_validado(config, caminho=FICHEIRO_INDICE):
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ErroProjeto("Ainda não há índice. Importa os PDF e executa python -m src.indice construir.")
    try:
        info = caminho.stat()
        indice, metadados = abrir_indice(caminho, info.st_mtime_ns, info.st_size)
        if metadados["corpus_sha256"] != sha256(CSV_CORPUS):
            raise ValueError("O corpus.csv mudou.")
        if (metadados["embeddings"], metadados["chunk_carateres"], metadados["sobreposicao"]) != (
            config.embeddings_modelo, config.chunk_carateres, config.sobreposicao,
        ):
            raise ValueError("A configuração do índice mudou.")
        fontes = selecionadas(ler_corpus())
        docs = {d["doc"]: d["sha256"] for d in metadados["documentos"]}
        if docs != {doc: f["sha256"] for doc, f in fontes.items()}:
            raise ValueError("Os documentos indexados diferem do corpus.")
        if not all(p["doc"] in docs for p in metadados["passagens"]):
            raise ValueError("O índice contém documentos excluídos.")
    except (OSError, ValueError, TypeError, IndexError, KeyError) as erro:
        raise ErroProjeto(f"Índice incompatível. Executa python -m src.indice construir. {erro}") from erro
    return indice, metadados


def recuperar(config, consulta, caminho=FICHEIRO_INDICE):
    indice, metadados = abrir_validado(config, caminho)
    if metadados["embeddings_digest"] != digest_modelo(config, config.embeddings_modelo):
        raise ErroProjeto("O modelo de embeddings mudou. Executa python -m src.indice construir.")
    consulta_vetor = matriz_normalizada(embeddings(config, [consulta]))
    if consulta_vetor.shape[1] != indice.d:
        raise ErroProjeto("A dimensão dos embeddings mudou. Reconstrói o índice.")
    pontuacoes, posicoes = indice.search(consulta_vetor, config.top_k)
    return [
        {**metadados["passagens"][int(posicao)], "similaridade": float(pontuacao)}
        for posicao, pontuacao in zip(posicoes[0], pontuacoes[0])
        if posicao >= 0
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="acao", required=True)
    build = sub.add_parser("construir", help="Extrair, criar embeddings e guardar o índice.")
    build.add_argument("--ocr", choices=["auto", "off", "por"], default="auto")
    build.add_argument("--lote", type=int, default=24)
    sub.add_parser("info", help="Listar documentos, páginas e passagens do índice.")
    parser.add_argument("--config", choices=EXPERIENCIAS, default="llm_cnn_rag")
    args = parser.parse_args()
    try:
        config = carregar_config(args.config)
        if args.acao == "construir":
            if args.lote < 1:
                parser.error("--lote deve ser pelo menos 1.")
            construir(config, opcao_ocr=args.ocr, lote=args.lote)
        else:
            _, dados = abrir_validado(config)
            print(f"{len(dados['passagens'])} passagens · {dados['embeddings']} · OCR: {dados['ocr']}")
            for doc in dados["documentos"]:
                print(f"{doc['doc']}: {doc['paginas']} páginas, {doc['passagens']} passagens, "
                      f"{len(doc['paginas_ocr'])} páginas com OCR, "
                      f"{len(doc['paginas_sem_texto'])} sem texto.")
    except (ErroProjeto, OSError, subprocess.SubprocessError) as erro:
        parser.exit(1, f"{erro}\n")


if __name__ == "__main__":
    main()
