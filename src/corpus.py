"""Importação dos PDF, verificação de versões e extração por página."""
import argparse
import csv
import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import httpx
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .apoio import ErroProjeto, RAIZ, agora, escrever_atomico, gravar_json, ler_json, sha256

CSV_CORPUS = RAIZ / "data/corpus.csv"
PASTA_PDF = RAIZ / "data/docs"
ESTADOS_INDICE = {"sim", "condicional"}
# Estes documentos do Grupo 2 são reservados mesmo que o CSV seja editado por engano.
HOLDOUT = {"D10", "D11", "D12"}
BRASIL = {"D07", "D08", "D13", "D14", "D15"}


def ler_corpus():
    fontes = {}
    hashes = set()
    with CSV_CORPUS.open(encoding="utf-8-sig", newline="") as ficheiro:
        for fonte in csv.DictReader(ficheiro):
            doc = fonte["doc_id"].strip()
            digest = fonte["sha256"].strip().lower()
            if not re.fullmatch(r"D\d+", doc) or doc in fontes:
                raise ErroProjeto(f"Identificador ausente, inválido ou repetido no corpus: {doc!r}")
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in hashes:
                raise ErroProjeto(f"SHA-256 inválido ou repetido no corpus: {doc}")
            if fonte["indexar"] not in ESTADOS_INDICE | {"não"}:
                raise ErroProjeto(f"Valor inesperado na coluna indexar de {doc}.")
            fontes[doc] = {**fonte, "doc_id": doc, "sha256": digest}
            hashes.add(digest)
    return fontes


def selecionadas(fontes):
    return {
        doc: f for doc, f in fontes.items()
        if f["indexar"] in ESTADOS_INDICE and doc not in HOLDOUT
    }


def guardar_pdf(fonte, dados, origem):
    doc = fonte["doc_id"]
    if hashlib.sha256(dados).hexdigest() != fonte["sha256"]:
        raise ErroProjeto(f"{doc}: o PDF não corresponde ao SHA-256 de data/corpus.csv.")
    destino = PASTA_PDF / f"{doc}.pdf"
    escrever_atomico(destino, lambda f: f.write(dados))
    metadados = PASTA_PDF / "origem.json"
    origens = ler_json(metadados) if metadados.exists() else {}
    origens[doc] = {"origem": origem, "registado_em": agora(), "sha256": fonte["sha256"]}
    gravar_json(metadados, origens)


def importar_zip(caminho):
    """O hash, e não o nome no ZIP, determina o identificador do documento."""
    fontes = ler_corpus()
    por_hash = {f["sha256"]: f for f in fontes.values()}
    importados = set()
    with ZipFile(caminho) as arquivo:
        for membro in arquivo.infolist():
            if membro.is_dir() or not membro.filename.lower().endswith(".pdf"):
                continue
            dados = arquivo.read(membro)
            fonte = por_hash.get(hashlib.sha256(dados).hexdigest())
            if fonte is None:
                print(f"PDF sem correspondência no corpus: {membro.filename}")
                continue
            guardar_pdf(fonte, dados, f"ZIP/{Path(membro.filename).name}")
            doc = fonte["doc_id"]
            importados.add(doc)
            print(f"{doc}: importado ({fonte['indexar']})")
    print(f"{len(importados)} PDF importados.")
    verificar(fontes, exigir_todos=False)


def descarregar():
    """Alternativa para quem só tem o corpus.csv e precisa de obter os originais."""
    falhas = []
    with httpx.Client(follow_redirects=True, timeout=60) as cliente:
        for doc, fonte in selecionadas(ler_corpus()).items():
            destino = PASTA_PDF / f"{doc}.pdf"
            if destino.is_file() and sha256(destino) == fonte["sha256"]:
                print(f"{doc}: já disponível.")
                continue
            try:
                resposta = cliente.get(fonte["url"])
                resposta.raise_for_status()
                guardar_pdf(fonte, resposta.content, str(resposta.url))
                print(f"{doc}: descarregado e verificado.")
            except (httpx.HTTPError, ErroProjeto) as erro:
                print(f"{doc}: {erro}")
                falhas.append(doc)
    if falhas:
        raise ErroProjeto(f"Usa o ZIP do grupo para os documentos: {', '.join(falhas)}.")


def verificar(fontes, exigir_todos=True):
    problemas = []
    for doc, fonte in fontes.items():
        pdf = PASTA_PDF / f"{doc}.pdf"
        if not pdf.is_file():
            problemas.append(f"{doc} em falta")
        elif sha256(pdf) != fonte["sha256"]:
            problemas.append(f"{doc} com SHA-256 diferente")
    if problemas:
        msg = " | ".join(problemas)
        if exigir_todos:
            raise ErroProjeto(f"{msg}. Importa o ZIP com python -m src.corpus importar.")
        print(msg)
    else:
        print(f"SHA-256 confirmado para {len(fontes)} PDF.")


def escolher_ocr(opcao):
    if opcao == "off":
        return None
    executavel = shutil.which("tesseract")
    if executavel:
        langs = subprocess.run(
            [executavel, "--list-langs"], capture_output=True, text=True, timeout=15, check=False,
        )
        if "por" in langs.stdout.splitlines():
            return executavel
    if opcao == "por":
        raise ErroProjeto("Instala o Tesseract com o idioma português (por) ou usa --ocr off.")
    return None


def limpar(texto):
    texto = texto.replace("\u00ad", "").replace("\x00", "")
    texto = re.sub(r"(\w)-\n(?=[a-zà-ÿ])", r"\1", texto)
    return "\n".join(" ".join(linha.split()) for linha in texto.splitlines()).strip()


def letras(texto):
    return sum(carater.isalpha() for carater in texto)


def extrair_passagens(config, ocr):
    fontes = selecionadas(ler_corpus())
    verificar(fontes)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_carateres, chunk_overlap=config.sobreposicao,
    )
    passagens, relatorio = [], []
    for doc, fonte in fontes.items():
        antes = len(passagens)
        paginas_sem_texto = []
        paginas_ocr = []
        carateres = 0
        with pymupdf.open(PASTA_PDF / f"{doc}.pdf") as pdf:
            if len(pdf) != int(fonte["paginas"]):
                raise ErroProjeto(f"{doc}: o número de páginas do PDF difere do corpus.csv.")
            for numero, pagina in enumerate(pdf, 1):
                texto = limpar(pagina.get_text(sort=True))
                if ocr and letras(texto) < 60:
                    imagem = pagina.get_pixmap(dpi=200)
                    proc = subprocess.run(
                        [ocr, "stdin", "stdout", "-l", "por"],
                        input=imagem.tobytes("png"), capture_output=True, timeout=90, check=True,
                    )
                    lido = limpar(proc.stdout.decode("utf-8"))
                    if letras(lido) > letras(texto):
                        texto = lido
                        paginas_ocr.append(numero)
                if letras(texto) < 35:
                    paginas_sem_texto.append(numero)
                    continue
                carateres += len(texto)
                for ordem, trecho in enumerate(splitter.split_text(texto), 1):
                    passagens.append({
                        "id": f"{doc}:{numero}:{ordem}", "doc": doc, "pagina": numero,
                        "data": fonte["data"], "titulo": fonte["titulo"],
                        "entidade": fonte["entidade"], "url": fonte["url"],
                        "contexto": "Brasil" if doc in BRASIL else "Portugal / orientação geral",
                        "texto": trecho,
                    })
            resumo = {
                "doc": doc, "sha256": fonte["sha256"], "paginas": len(pdf),
                "carateres": carateres, "passagens": len(passagens) - antes,
                "paginas_ocr": paginas_ocr, "paginas_sem_texto": paginas_sem_texto,
            }
        relatorio.append(resumo)
        print(f"{doc}: {resumo['paginas']} páginas, {resumo['passagens']} passagens "
              f"({len(paginas_ocr)} páginas com OCR, {len(paginas_sem_texto)} sem texto).", flush=True)
    return passagens, relatorio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="acao", required=True)
    importar = sub.add_parser("importar", help="Importar o ZIP original dos PDF, por hash.")
    importar.add_argument("zip", type=Path)
    sub.add_parser("descarregar", help="Descarregar os candidatos diretamente das fontes.")
    sub.add_parser("verificar", help="Confirmar os hashes dos documentos a indexar.")
    args = parser.parse_args()
    try:
        if args.acao == "importar":
            importar_zip(args.zip)
        elif args.acao == "descarregar":
            descarregar()
        else:
            verificar(selecionadas(ler_corpus()))
    except (ErroProjeto, OSError, BadZipFile) as erro:
        parser.exit(1, f"{erro}\n")


if __name__ == "__main__":
    main()
