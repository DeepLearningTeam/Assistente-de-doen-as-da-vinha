"""Validação de fotografias e integração do classificador CNN do grupo."""
import base64
import importlib
import math
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, PositiveInt, model_validator

from .apoio import ErroProjeto, RAIZ

CLASSES = {
    "saudavel": "Saudável",
    "esca": "Esca",
    "leaf_blight": "Mancha-das-folhas (leaf blight)",
    "black_rot": "Podridão-negra (black rot)",
}
Classe = Literal["saudavel", "esca", "leaf_blight", "black_rot"]
LIMITE_BYTES = 12_000_000


class FotoInvalida(ErroProjeto):
    """Ficheiro que não pode ser usado como fotografia."""


class Visual(BaseModel):
    motor: Literal["CNN", "VLM"]
    estado: Literal["ok", "incerta", "rejeitada"]
    classe: Classe | None = None
    confianca: float | None = Field(default=None, ge=0, le=1)
    probabilidades: dict[str, float] = Field(default_factory=dict)
    descricao: str = ""
    mapa_atencao: str | None = None


class ConfigCNN(BaseModel):
    backend: Literal["onnx", "python"]
    checkpoint: str
    adaptador: str
    largura: PositiveInt
    altura: PositiveInt
    redimensionamento: Literal["direto", "centro"]
    layout: Literal["NCHW", "NHWC"]
    escala: float = Field(gt=0)
    media: tuple[float, float, float]
    desvio: tuple[float, float, float]
    saida: Literal["logits", "probabilidades"]
    nome_saida: str | None = None
    classes: list[str]
    limiar: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validar_config(self):
        if set(self.classes) not in (set(CLASSES), set(CLASSES) | {"outra"}):
            raise ValueError("As classes devem corresponder às quatro classes do FieldVitis.")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("A ordem das classes da CNN contém repetições.")
        if any(d <= 0 or not math.isfinite(d) for d in self.desvio):
            raise ValueError("Os desvios de normalização devem ser positivos e finitos.")
        return self


def ler_imagem(caminho):
    caminho = Path(caminho)
    try:
        if caminho.stat().st_size > LIMITE_BYTES:
            raise FotoInvalida("A fotografia excede o limite de 12 MB.")
        with Image.open(caminho) as imagem:
            if imagem.format not in {"JPEG", "PNG"}:
                raise FotoInvalida("Seleciona uma fotografia em JPG ou PNG.")
            if imagem.width * imagem.height > 40_000_000:
                raise FotoInvalida("A fotografia é demasiado grande. Reduz a resolução.")
            return ImageOps.exif_transpose(imagem).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as erro:
        raise FotoInvalida("Não foi possível abrir a fotografia. Seleciona um JPG ou PNG válido.") from erro


def codificar_imagem(imagem, lado_maximo=1400):
    """JPEG em memória, sem metadados EXIF nem localização da fotografia."""
    copia = imagem.convert("RGB")
    copia.thumbnail((lado_maximo, lado_maximo), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    copia.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def descodificar_imagem(texto):
    with Image.open(BytesIO(base64.b64decode(texto, validate=True))) as imagem:
        return imagem.convert("RGB")


def config_cnn():
    try:
        texto = (RAIZ / "configs/cnn.yaml").read_text(encoding="utf-8")
        return ConfigCNN.model_validate(yaml.safe_load(texto))
    except (OSError, ValueError, yaml.YAMLError) as erro:
        raise ErroProjeto(f"Revê configs/cnn.yaml. {erro}") from erro


@lru_cache(maxsize=4)
def sessao_onnx(caminho, modificacao_ns):
    try:
        import onnxruntime
    except ImportError as erro:
        raise ErroProjeto("Instala a dependência da CNN com: python -m pip install onnxruntime==1.31.1") from erro
    try:
        return onnxruntime.InferenceSession(str(caminho), providers=["CPUExecutionProvider"])
    except Exception as erro:
        raise ErroProjeto(f"Não foi possível abrir a CNN ONNX em {caminho.name}. {erro}") from erro


def prever_onnx(imagem, caminho, config):
    tamanho = (config.largura, config.altura)
    if config.redimensionamento == "centro":
        imagem = ImageOps.fit(imagem, tamanho, Image.Resampling.BILINEAR)
    else:
        imagem = imagem.resize(tamanho, Image.Resampling.BILINEAR)
    tensor = np.asarray(imagem, dtype=np.float32) / config.escala
    tensor = (tensor - np.asarray(config.media, dtype=np.float32)) / np.asarray(config.desvio, dtype=np.float32)
    if config.layout == "NCHW":
        tensor = tensor.transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor[None, ...])
    sessao = sessao_onnx(caminho, caminho.stat().st_mtime_ns)
    try:
        saidas = [config.nome_saida] if config.nome_saida else None
        valores = np.asarray(sessao.run(saidas, {sessao.get_inputs()[0].name: tensor})[0]).reshape(-1)
    except Exception as erro:
        raise ErroProjeto(f"A entrada ou saída da CNN não corresponde a configs/cnn.yaml. {erro}") from erro
    if len(valores) != len(config.classes) or not np.isfinite(valores).all():
        raise ErroProjeto("A CNN devolveu uma saída incompatível com a lista de classes.")
    if config.saida == "logits":
        # Softmax transforma os valores do modelo em probabilidades com soma igual a 1.
        valores = np.exp(valores.astype(np.float64) - valores.max())
        valores /= valores.sum()
    return {"probabilidades": dict(zip(config.classes, map(float, valores)))}


def prever_python(imagem, caminho, config):
    modulo, separador, funcao = config.adaptador.partition(":")
    if not separador:
        raise ErroProjeto("Indica o adaptador Python como modulo:funcao em configs/cnn.yaml.")
    try:
        prever = getattr(importlib.import_module(modulo), funcao)
        return prever(imagem, caminho)
    except NotImplementedError as erro:
        raise ErroProjeto("Preenche src/cnn_colega.py com a inferência do teu colega.") from erro
    except Exception as erro:
        raise ErroProjeto(f"O adaptador da CNN falhou. {erro}") from erro


def analisar_cnn(imagem, checkpoint=None):
    config = config_cnn()
    caminho = Path(checkpoint) if checkpoint else RAIZ / config.checkpoint
    if not caminho.is_file():
        raise ErroProjeto(
            f"A CNN ainda não está ligada. Coloca o modelo em {caminho.name} "
            "conforme runs/checkpoints/README.md ou escolhe um modo VLM."
        )
    if config.backend == "onnx":
        resultado = prever_onnx(imagem, caminho, config)
    else:
        resultado = prever_python(imagem, caminho, config)
    if not isinstance(resultado, dict):
        raise ErroProjeto("O adaptador CNN deve devolver um dicionário.")
    if resultado.get("e_folha") is False:
        return Visual(motor="CNN", estado="rejeitada", descricao="A CNN rejeitou a fotografia como folha de videira.")

    try:
        probabilidades = {k: float(v) for k, v in resultado["probabilidades"].items()}
        if set(probabilidades) not in (set(CLASSES), set(CLASSES) | {"outra"}):
            raise ValueError("classes inesperadas")
        if any(not 0 <= p <= 1 for p in probabilidades.values()) or not math.isclose(
            sum(probabilidades.values()), 1.0, abs_tol=0.01,
        ):
            raise ValueError("as probabilidades devem somar 1")
    except (KeyError, TypeError, ValueError, AttributeError) as erro:
        raise ErroProjeto(f"Probabilidades inválidas no adaptador CNN. {erro}") from erro

    classe = max(probabilidades, key=probabilidades.get)
    confianca = probabilidades[classe]
    if classe == "outra":
        return Visual(motor="CNN", estado="rejeitada", descricao="A CNN identificou uma imagem fora das classes da videira.")
    estado = "ok" if confianca >= config.limiar else "incerta"
    descricao = resultado.get("descricao") or "Classificação baseada nos padrões da folha."
    mapa = resultado.get("mapa_atencao")
    return Visual(
        motor="CNN", estado=estado, classe=classe, confianca=confianca,
        probabilidades=probabilidades, descricao=str(descricao)[:500],
        mapa_atencao=codificar_imagem(mapa) if isinstance(mapa, Image.Image) else None,
    )
