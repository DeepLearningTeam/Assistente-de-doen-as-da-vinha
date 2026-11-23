"""Operações pequenas partilhadas pela aplicação e pelos comandos."""
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_IMAGENS = RAIZ / "data/imagens"
FICHEIRO_SPLITS = RAIZ / "data/splits.json"


class ErroProjeto(Exception):
    """Erro com uma mensagem que pode ser mostrada ao utilizador."""


def agora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(caminho):
    with Path(caminho).open("rb") as ficheiro:
        return hashlib.file_digest(ficheiro, "sha256").hexdigest()


def escrever_atomico(caminho, escrever):
    """Troca o ficheiro antigo pelo novo só depois de a escrita terminar."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = None
    try:
        with tempfile.NamedTemporaryFile(dir=caminho.parent, delete=False) as ficheiro:
            temporario = Path(ficheiro.name)
            escrever(ficheiro)
        os.replace(temporario, caminho)
    finally:
        if temporario is not None:
            temporario.unlink(missing_ok=True)


def gravar_json(caminho, dados):
    texto = json.dumps(dados, ensure_ascii=False, indent=2) + "\n"
    escrever_atomico(caminho, lambda ficheiro: ficheiro.write(texto.encode("utf-8")))


def ler_json(caminho):
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def nova_verificacao(nome):
    """Cria um relatório local com data, sem substituir verificações anteriores."""
    instante = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    pasta = RAIZ / "runs/verificacao" / f"{instante}-{nome}"
    pasta.mkdir(parents=True, exist_ok=False)
    gravar_json(pasta / "execucao.json", {
        "tipo": "verificacao_funcional", "criado": agora(),
        "python": platform.python_version(), "plataforma": platform.platform(),
        "protocolo_sha256": sha256(RAIZ / "PROTOCOLO.md"),
        "codigo_sha256": {p.name: sha256(p) for p in sorted((RAIZ / "src").glob("*.py"))},
        "configuracoes_sha256": {p.name: sha256(p) for p in sorted((RAIZ / "configs").glob("*.yaml"))},
        "testes_sha256": {p.name: sha256(p) for p in sorted((RAIZ / "tests").glob("test_*.py"))},
    })
    return pasta
