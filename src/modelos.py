"""Clientes LangChain para chat e embeddings no Ollama local."""
from functools import lru_cache

import httpx
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ollama import Client, ResponseError
from pydantic import ValidationError

from .apoio import ErroProjeto


class ErroFormato(ErroProjeto):
    """O modelo respondeu, mas o programa não conseguiu interpretar a saída."""


def erro_ollama(erro, modelo, url):
    if isinstance(erro, ResponseError):
        if erro.status_code == 404:
            return ErroProjeto(f"Instala o modelo no Ollama com: ollama pull {modelo}")
        return ErroProjeto(f"O Ollama devolveu um erro ao usar {modelo}. Consulta o terminal.")
    if isinstance(erro, httpx.TimeoutException):
        return ErroProjeto(f"O modelo {modelo} excedeu o tempo de espera. Tenta novamente.")
    return ErroProjeto(f"Não foi possível ligar ao Ollama em {url}. Abre o Ollama ou executa ollama serve.")


@lru_cache(maxsize=24)
def cliente_chat(url, modelo, timeout, contexto, temperatura, seed, limite):
    return ChatOllama(
        base_url=url,
        model=modelo,
        temperature=temperatura,
        seed=seed,
        num_ctx=contexto,
        num_predict=limite,
        reasoning=False if modelo.startswith("qwen3") else None,
        client_kwargs={"timeout": timeout, "trust_env": False},
    )


def conversar(config, mensagens, esquema, *, limite=None, deterministico=False, registo=None, etapa="resposta"):
    """O esquema JSON permite validar a decisão e as citações em código."""
    modelo = config.modelo
    cliente = cliente_chat(
        config.ollama_url, modelo, config.timeout_segundos, config.contexto_tokens,
        0.0 if deterministico else config.temperatura,
        config.seed, limite or config.max_resposta_tokens,
    )
    if registo is not None:
        # Para a avaliação, conservar o texto exato do prompt sem replicar bytes da imagem.
        partes = []
        for mensagem in mensagens:
            conteudo = mensagem.content
            if isinstance(conteudo, list):
                conteudo = [p.get("text", "[fotografia]") for p in conteudo]
            partes.append({"papel": mensagem.type, "conteudo": conteudo})
        registo.append({"etapa": etapa, "modelo": modelo, "mensagens": partes})
    try:
        mensagem = cliente.bind(format=esquema.model_json_schema()).invoke(mensagens)
    except (ConnectionError, httpx.HTTPError, ResponseError) as erro:
        raise erro_ollama(erro, modelo, config.ollama_url) from erro
    resposta = mensagem.content
    if registo is not None:
        registo[-1]["saida_modelo"] = resposta
    try:
        return esquema.model_validate_json(resposta)
    except (ValidationError, TypeError) as erro:
        raise ErroFormato("O modelo devolveu uma resposta incompleta. Tenta reformular a pergunta.") from erro


@lru_cache(maxsize=8)
def cliente_embeddings(url, modelo, timeout):
    return OllamaEmbeddings(
        base_url=url, model=modelo, client_kwargs={"timeout": timeout, "trust_env": False},
    )


def embeddings(config, textos):
    cliente = cliente_embeddings(config.ollama_url, config.embeddings_modelo, config.timeout_segundos)
    try:
        return cliente.embed_documents(textos)
    except (ConnectionError, httpx.HTTPError, ResponseError) as erro:
        raise erro_ollama(erro, config.embeddings_modelo, config.ollama_url) from erro


def digest_modelo(config, modelo):
    """O nome de um modelo é uma etiqueta. O digest identifica os pesos locais."""
    try:
        cliente = Client(host=config.ollama_url, timeout=config.timeout_segundos, trust_env=False)
        modelos = cliente.list().models
    except (ConnectionError, httpx.HTTPError, ResponseError) as erro:
        raise erro_ollama(erro, modelo, config.ollama_url) from erro
    nome = modelo if ":" in modelo.rsplit("/", 1)[-1] else f"{modelo}:latest"
    for item in modelos:
        if item.model == nome:
            return item.digest
    raise ErroProjeto(f"O modelo {modelo} não está instalado. Executa: ollama pull {modelo}")
