"""Os nomes e parâmetros das experiências vivem em configs/."""
import os
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from .apoio import ErroProjeto, RAIZ

MODOS = {
    "llm_cnn_rag": "LLM + CNN + RAG",
    "llm_cnn": "LLM + CNN",
    "vlm": "VLM",
    "vlm_rag": "VLM + RAG",
}
EXPERIENCIAS = {**MODOS, "llm_rag_sem_limiar": "LLM + RAG sem limiar"}


class Config(BaseModel):
    gerador: Literal["llm", "vlm"]
    rag: bool
    recusa_reforcada: bool
    llm_modelo: str
    vlm_modelo: str
    embeddings_modelo: str
    ollama_url: str
    timeout_segundos: int = Field(gt=0)
    contexto_tokens: int = Field(ge=2048)
    max_resposta_tokens: int = Field(gt=0)
    temperatura: float = Field(ge=0, le=2)
    seed: int
    historico_turnos: int = Field(gt=0)
    top_k: int = Field(ge=1, le=10)
    chunk_carateres: int = Field(gt=0)
    sobreposicao: int = Field(ge=0)
    reranker_modelo: str = "BAAI/bge-reranker-v2-m3"
    pesquisa: Literal["denso", "hibrida"] = "hibrida"
    k_candidatos: int = Field(default=30, ge=1)
    limiar_reranker: float = Field(default=0.60, ge=0, le=1)
    qdrant_colecao: str = "videira_fitossanidade"
    confianca_visual_minima: float = Field(ge=0, le=1)
    prompt: str

    @property
    def modelo(self):
        return self.llm_modelo if self.gerador == "llm" else self.vlm_modelo

    @model_validator(mode="after")
    def validar_chunks(self):
        if self.sobreposicao >= self.chunk_carateres:
            raise ValueError("A sobreposição deve ser menor do que o tamanho do chunk.")
        return self


def carregar_config(modo="llm_cnn_rag", **alteracoes):
    # Caminho explícito: o arranque não depende da pasta onde o terminal estava.
    # As variáveis da sessão têm prioridade sobre o ficheiro local.
    load_dotenv(RAIZ / ".env", override=False)
    if modo not in EXPERIENCIAS:
        raise ErroProjeto(f"Modo desconhecido: {modo}")
    pasta = RAIZ / "configs"
    try:
        base = yaml.safe_load((pasta / "base.yaml").read_text(encoding="utf-8"))
        via = yaml.safe_load((pasta / f"{modo}.yaml").read_text(encoding="utf-8"))
        valores = {**base, **via, **alteracoes}
        valores["ollama_url"] = os.environ.get("OLLAMA_BASE_URL", valores["ollama_url"])
        valores["qdrant_colecao"] = os.environ.get("QDRANT_COLECAO", valores["qdrant_colecao"])
        return Config.model_validate(valores)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as erro:
        raise ErroProjeto(f"Verificar os YAML em configs/. {erro}") from erro
