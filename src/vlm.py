"""Inferência visual zero-shot com Qwen2.5-VL através do Ollama.

O VLM produz apenas interpretação visual/classificação.
A resposta técnica fundamentada em documentos continua a ser da
responsabilidade do LLM textual + RAG.
"""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.prompts import VLM_CLASSIFICATION_PROMPT

CLASSES = {
    "saudavel",
    "esca",
    "leaf_blight",
    "black_rot",
    "inconclusivo",
}


class Qwen25VL:
    def __init__(
        self,
        model_id: str = "qwen2.5vl:3b-q4_K_M",
        base_url: str | None = None,
        timeout: int = 600,
    ):
        load_dotenv()

        if not model_id or model_id.startswith(("PREENCHER", "ESCOLHER")):
            raise RuntimeError("Define VLM_MODEL no ficheiro .env.")

        self.model_id = model_id
        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            )
        ).rstrip("/")

        self.timeout = timeout
        self.session = requests.Session()

    def analisar(
        self,
        imagem: str | Path,
        pergunta: str = ("Classifica a doença provável nesta folha de videira."),
        max_new_tokens: int = 180,
    ) -> dict[str, Any]:

        caminho = Path(imagem)

        if not caminho.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {caminho}")

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens tem de ser positivo.")

        # Converter a imagem para Base64, formato aceite pela API Ollama.
        imagem_base64 = base64.b64encode(caminho.read_bytes()).decode("utf-8")

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": VLM_CLASSIFICATION_PROMPT,
                },
                {
                    "role": "user",
                    "content": pergunta,
                    "images": [imagem_base64],
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": max_new_tokens,
            },
        }

        inicio = time.perf_counter()

        try:
            resposta_http = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resposta_http.raise_for_status()

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Não foi possível ligar ao Ollama. "
                "Confirma que o Ollama está aberto e disponível em "
                f"{self.base_url}."
            ) from exc

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"O Ollama excedeu o limite de {self.timeout} segundos."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            detalhe = resposta_http.text[:1000]

            raise RuntimeError(
                f"Erro HTTP {resposta_http.status_code} do Ollama. "
                f"Confirma que o modelo '{self.model_id}' está instalado "
                f"(comando: ollama list).\nDetalhes: {detalhe}"
            ) from exc

        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Erro na comunicação com o Ollama: {exc}") from exc

        latencia = time.perf_counter() - inicio

        try:
            dados = resposta_http.json()
        except ValueError as exc:
            raise RuntimeError(
                "O Ollama devolveu uma resposta que não é JSON válido."
            ) from exc

        resposta_texto = dados.get("message", {}).get("content", "").strip()

        if not resposta_texto:
            raise RuntimeError("O Ollama não devolveu conteúdo de texto.")

        classe = _extrair_classe(resposta_texto)
        descricao = _extrair_linha(resposta_texto, "DESCRICAO")
        limitacoes = _extrair_linha(resposta_texto, "LIMITACOES")

        return {
            "classe": classe,
            "descricao_visual": descricao,
            "limitacoes": limitacoes,
            "texto_bruto": resposta_texto,
            "model_id": dados.get("model", self.model_id),
            "backend": "ollama",
            "device": "ollama_local",
            "dtype": "gerido pelo Ollama",
            "latencia_s": round(latencia, 4),
            "confidence": None,
            "imagem": str(caminho),
            "tokens_prompt": dados.get("prompt_eval_count"),
            "tokens_resposta": dados.get("eval_count"),
        }


def _extrair_linha(texto: str, chave: str) -> str:
    match = re.search(
        rf"(?im)^\s*{re.escape(chave)}\s*:\s*(.+)$",
        texto,
    )
    return match.group(1).strip() if match else ""


def _extrair_classe(texto: str) -> str:
    valor = _extrair_linha(texto, "CLASSE").lower().replace(" ", "_").replace("-", "_")

    aliases = {
        "healthy": "saudavel",
        "saudável": "saudavel",
        "leafblight": "leaf_blight",
        "blackrot": "black_rot",
        "inconclusiva": "inconclusivo",
    }

    valor = aliases.get(valor, valor)

    return valor if valor in CLASSES else "inconclusivo"


def carregar_vlm(model_id: str | None = None) -> Qwen25VL:
    load_dotenv()

    escolhido = model_id or os.getenv("VLM_MODEL") or "qwen2.5vl:3b-q4_K_M"

    base_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    return Qwen25VL(
        model_id=escolhido,
        base_url=base_url,
    )
