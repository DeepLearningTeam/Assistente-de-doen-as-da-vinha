"""Interface para a CNN de classificação, a integrar com o modelo do F2.

Contrato proposto: devolver pelo menos `classe` e `probabilidades`.
Este módulo não contém um modelo inventado: ligar aqui o modelo e o pré-processamento reais.
"""
from __future__ import annotations
from typing import Any
from PIL import Image


def classificar_folha(imagem: Image.Image) -> dict[str, Any]:
    """Classifica uma folha usando a CNN real do grupo.

    TODO (em coordenação com F2): carregar o modelo treinado, aplicar exatamente o
    pré-processamento usado no treino e devolver as probabilidades por classe.

    Exemplo de formato de retorno, NÃO é resultado real:
        {"classe": "esca", "probabilidades": {"saudavel": 0.1, "esca": 0.7, ...}}
    """
    raise NotImplementedError(
        "Integrar a CNN treinada pelo F2 e verificar o pré-processamento do modelo."
    )
