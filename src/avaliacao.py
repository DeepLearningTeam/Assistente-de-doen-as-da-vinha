"""Registo reproduzível de execuções. A fidelidade continua a ser anotada por pessoas."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def guardar_execucao(
    experimento: str,
    pergunta_id: str,
    repeticao: int,
    entrada: dict[str, Any],
    saida: dict[str, Any],
    configuracao: dict[str, Any],
    diretorio_runs: str | Path = "runs",
) -> Path:
    """Guarda um JSON por pergunta/repetição, sem guardar imagens por defeito."""
    if repeticao < 1:
        raise ValueError("repeticao deve começar em 1")
    pasta = Path(diretorio_runs) / experimento
    pasta.mkdir(parents=True, exist_ok=True)
    registo = {
        "experimento": experimento,
        "pergunta_id": pergunta_id,
        "repeticao": repeticao,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "entrada": entrada,
        "saida": saida,
        "configuracao": configuracao,
    }
    destino = pasta / f"{pergunta_id}_rep{repeticao}.json"
    destino.write_text(json.dumps(registo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return destino
