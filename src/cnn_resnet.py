"""Inferência adaptada de inference/predict.py da implementação original (ResNet50, seed 1).

Mantém o pré-processamento e os pesos originais. A aplicação fornece uma PIL;
a ordem dos neurónios continua a vir do splits.json original do treino.
"""
import json
from functools import lru_cache
from threading import Lock

import numpy as np
from PIL import Image, ImageOps

from .apoio import ErroProjeto, RAIZ

SPLITS_ORIGINAIS = RAIZ / "data/originais/cnn/splits.json"
MAPEAMENTO = {
    "Black_rot": "black_rot",
    "Esca": "esca",
    "Healthy": "saudavel",
    "Leaf_blight": "leaf_blight",
    "Outro": "outro",
}
IMG_SIZE = 224
# Valor fornecido com o modelo, escolhido na validação da ResNet50 seed 1.
LIMIAR_CONFIANCA = 0.58
_CARREGAMENTO = Lock()


def obter_keras():
    # Importação tardia: texto e VLM funcionam sem TensorFlow/Keras.
    try:
        import keras
        if keras.backend.backend() != "tensorflow":
            raise ErroProjeto("Esta CNN foi verificada com KERAS_BACKEND=tensorflow.")
        return keras
    except (ImportError, OSError) as erro:
        raise ErroProjeto(
            "A CNN precisa de TensorFlow/Keras. Na raiz do projeto executar "
            r".\.venv\Scripts\python.exe -m pip install -r requirements.txt"
        ) from erro


def carregar_classes():
    classes = json.loads(SPLITS_ORIGINAIS.read_text(encoding="utf-8"))["classes"]
    if classes != list(MAPEAMENTO):
        raise ErroProjeto("A ordem das classes difere do contrato ResNet50 recebido.")
    return classes


@lru_cache(maxsize=1)
def _carregar_modelo(caminho, modificacao_ns):
    # compile=False dispensa o otimizador de treino; não altera os pesos.
    modelo = obter_keras().models.load_model(caminho, compile=False)
    if tuple(modelo.input_shape) != (None, IMG_SIZE, IMG_SIZE, 3) or tuple(modelo.output_shape) != (None, len(MAPEAMENTO)):
        raise ErroProjeto("Esperava entrada (None,224,224,3) e saída (None,5).")
    if modelo.layers[-1].get_config().get("activation") != "softmax":
        raise ErroProjeto("Este adaptador espera probabilidades softmax já calculadas pelo modelo.")
    return modelo


def carregar_modelo(caminho):
    if not caminho.is_file():
        raise ErroProjeto(f"Falta o modelo CNN: {caminho.name}. Consultar runs/checkpoints/README.md.")
    # Evita dois carregamentos simultâneos no primeiro pedido.
    with _CARREGAMENTO:
        return _carregar_modelo(str(caminho.resolve()), caminho.stat().st_mtime_ns)


def preparar_imagem(imagem):
    imagem = ImageOps.exif_transpose(imagem).convert("RGB")
    # Preserva a proporção e completa a imagem com barras pretas, como no novo treino.
    imagem = ImageOps.pad(imagem, (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR, color=(0, 0, 0))
    tensor = np.asarray(imagem, dtype=np.uint8).astype("float32")[None, ...]
    # RGB -> BGR e subtração das médias ImageNet; NÃO dividir por 255.
    return obter_keras().applications.resnet50.preprocess_input(tensor)


def prever(imagem, checkpoint):
    modelo = carregar_modelo(checkpoint)
    classes = carregar_classes()
    # training=False desativa augmentation e dropout, como model.predict da implementação original.
    valores = np.asarray(modelo(preparar_imagem(imagem), training=False))
    if valores.shape != (1, len(classes)) or not np.isfinite(valores).all():
        raise ErroProjeto("A CNN devolveu uma forma inesperada ou valores não finitos.")
    probabilidades = valores[0]
    if np.any(probabilidades < 0) or np.any(probabilidades > 1) or not np.isclose(probabilidades.sum(), 1, atol=1e-5):
        raise ErroProjeto("A saída da CNN não corresponde a probabilidades softmax.")
    indice = int(np.argmax(probabilidades))
    recognized = classes[indice] != "Outro" and float(probabilidades[indice]) >= LIMIAR_CONFIANCA
    return {
        "probabilidades": {MAPEAMENTO[c]: float(p) for c, p in zip(classes, probabilidades)},
        "recognized": recognized,
        "descricao": (
            "Hipótese entre as quatro classes de folhas do FieldVitis, com rejeição "
            "por Outro ou baixa confiança. As probabilidades softmax não são certeza de diagnóstico."
        ),
    }
