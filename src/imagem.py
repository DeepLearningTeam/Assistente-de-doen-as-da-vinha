"""Validação de fotografias e integração do classificador CNN do grupo."""
import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Literal

import yaml
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator

from .apoio import ErroProjeto, RAIZ
from .cnn_resnet import prever

CLASSES = {
    "saudavel": "Saudável",
    "esca": "Esca",
    "leaf_blight": "Mancha-das-folhas (leaf blight)",
    "black_rot": "Podridão-negra (black rot)",
}
Classe = Literal["saudavel", "esca", "leaf_blight", "black_rot"]
# Outro é uma saída técnica da CNN, não uma hipótese de doença nem uma classe do VLM.
ClasseCNN = Classe | Literal["outro"]
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
    model_config = {"extra": "forbid"}

    checkpoint: str
    seed_treino: int
    classes: list[ClasseCNN]

    @model_validator(mode="after")
    def validar_config(self):
        if set(self.classes) != set(CLASSES) | {"outro"}:
            raise ValueError("As classes devem corresponder ao FieldVitis e à classe Outro.")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("A ordem das classes da CNN contém repetições.")
        return self


def ler_imagem(caminho):
    caminho = Path(caminho)
    try:
        if caminho.stat().st_size > LIMITE_BYTES:
            raise FotoInvalida("A fotografia excede o limite de 12 MB.")
        with Image.open(caminho) as imagem:
            if imagem.format not in {"JPEG", "PNG"}:
                raise FotoInvalida("Selecionar uma fotografia em JPG ou PNG.")
            if imagem.width * imagem.height > 40_000_000:
                raise FotoInvalida("A fotografia é demasiado grande. Reduzir a resolução.")
            return ImageOps.exif_transpose(imagem).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as erro:
        raise FotoInvalida("Não foi possível abrir a fotografia. Selecionar um JPG ou PNG válido.") from erro


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
        raise ErroProjeto(f"Verificar configs/cnn.yaml. {erro}") from erro


def analisar_cnn(imagem, checkpoint=None):
    config = config_cnn()
    caminho = Path(checkpoint) if checkpoint else RAIZ / config.checkpoint
    if not caminho.is_file():
        raise ErroProjeto(
            f"A CNN ainda não está ligada. Colocar o modelo em {caminho.name} "
            "conforme runs/checkpoints/README.md ou selecionar um modo VLM."
        )
    resultado = prever(imagem, caminho)
    if not isinstance(resultado, dict):
        raise ErroProjeto("O adaptador CNN deve devolver um dicionário.")
    try:
        probabilidades = {k: float(v) for k, v in resultado["probabilidades"].items()}
        if set(probabilidades) != set(config.classes):
            raise ValueError("classes inesperadas")
        if any(not 0 <= p <= 1 for p in probabilidades.values()) or not math.isclose(
            sum(probabilidades.values()), 1.0, abs_tol=0.01,
        ):
            raise ValueError("as probabilidades devem somar 1")
    except (KeyError, TypeError, ValueError, AttributeError) as erro:
        raise ErroProjeto(f"Probabilidades inválidas no adaptador CNN. {erro}") from erro

    classe = max(probabilidades, key=probabilidades.get)
    confianca = probabilidades[classe]
    if not isinstance(resultado.get("recognized"), bool):
        raise ErroProjeto("O adaptador CNN deve indicar recognized como verdadeiro ou falso.")
    if not resultado["recognized"]:
        return Visual(
            motor="CNN", estado="rejeitada", confianca=confianca,
            probabilidades=probabilidades,
            descricao="A imagem não foi reconhecida como folha de videira pela CNN.",
        )
    descricao = resultado.get("descricao") or "Classificação baseada nos padrões da folha."
    mapa = resultado.get("mapa_atencao")
    return Visual(
        motor="CNN", estado="ok", classe=classe, confianca=confianca,
        probabilidades=probabilidades, descricao=str(descricao)[:500],
        mapa_atencao=codificar_imagem(mapa) if isinstance(mapa, Image.Image) else None,
    )
