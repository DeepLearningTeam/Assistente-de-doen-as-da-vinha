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


def gradcam(modelo, tensor, indice):
    """Grad-CAM da classe prevista, sem modificar a softmax ou os pesos em cache."""
    import tensorflow as tf

    keras = obter_keras()
    base = modelo.get_layer("resnet50")
    extrator = keras.Model(base.inputs, [base.get_layer("conv5_block3_out").output, base.output])
    entrada = modelo.layers[1](tf.convert_to_tensor(tensor), training=False)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(entrada)
        ativacoes, vetor = extrator([entrada], training=False)
        vetor = modelo.layers[-2](vetor, training=False)  # Dropout desativado.
        cabeca = modelo.layers[-1]
        # Usar o score anterior à softmax apenas neste cálculo auxiliar.
        scores = tf.matmul(vetor, cabeca.kernel) + cabeca.bias
        alvo = scores[:, indice]
    gradientes = tape.gradient(alvo, ativacoes)
    if gradientes is None:
        raise ErroProjeto("Não foi possível calcular os gradientes do mapa da CNN.")
    pesos = tf.reduce_mean(gradientes, axis=(0, 1, 2))
    mapa = tf.nn.relu(tf.reduce_sum(ativacoes[0] * pesos, axis=-1))
    # Um mapa sem contribuição positiva mantém-se a zero, sem divisões por zero.
    return tf.math.divide_no_nan(mapa, tf.reduce_max(mapa)).numpy()


def sobrepor_gradcam(imagem, mapa):
    """Retira o padding do mapa e sobrepõe-o à fotografia com orientação EXIF corrigida."""
    original = ImageOps.exif_transpose(imagem).convert("RGB")
    tamanho = (IMG_SIZE, IMG_SIZE)
    conteudo = ImageOps.contain(original, tamanho, Image.Resampling.BILINEAR)
    # Reproduzir os arredondamentos do ImageOps.pad, incluindo margens ímpares.
    esquerda = round((IMG_SIZE - conteudo.width) / 2)
    topo = round((IMG_SIZE - conteudo.height) / 2)
    mapa = Image.fromarray(np.asarray(mapa, dtype=np.float32)).resize(tamanho, Image.Resampling.BILINEAR)
    mapa = mapa.crop((esquerda, topo, esquerda + conteudo.width, topo + conteudo.height))
    intensidade = mapa.resize(original.size, Image.Resampling.BILINEAR).point(lambda v: v * 255).convert("L")
    cores = ImageOps.colorize(intensidade, "yellow", "red")
    # As zonas de contribuição nula ficam transparentes.
    mascara = intensidade.point(lambda v: round(v * 0.45))
    return Image.composite(cores, original, mascara)


def prever(imagem, checkpoint):
    modelo = carregar_modelo(checkpoint)
    classes = carregar_classes()
    # training=False desativa augmentation e dropout, como model.predict da implementação original.
    tensor = preparar_imagem(imagem)
    valores = np.asarray(modelo(tensor, training=False))
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
        "mapa_atencao": sobrepor_gradcam(imagem, gradcam(modelo, tensor, indice)) if recognized else None,
        "descricao": (
            "Hipótese entre as quatro classes de folhas do FieldVitis, com rejeição "
            "por Outro ou baixa confiança. As probabilidades softmax não são certeza de diagnóstico."
        ),
    }
