"""Interface comum para experimentar um VLM com Transformers.

A implementação concreta depende do modelo selecionado (p.ex., Qwen2.5-VL-3B ou
SmolVLM), do processador e dos recursos disponíveis. Escolher e testar o modelo antes
 de fixar a chamada de inferência.
"""
from __future__ import annotations
from typing import Any
from PIL import Image


def analisar_folha_vlm(
    pergunta: str,
    imagem: Image.Image,
    contexto_documental: str = "",
    model_id: str | None = None,
) -> dict[str, Any]:
    """Devolve análise visual e, na arquitetura final, resposta apoiada em documentos.

    TODO: carregar o VLM escolhido através de Transformers, preparar a imagem e o prompt,
    gerar texto e registar model_id, tempo e configurações. Confirmar o formato específico
    requerido pelo processor do modelo escolhido. O modelo não deve inventar citações:
    passar apenas as fontes recuperadas e verificar as referências depois.
    """
    raise NotImplementedError(
        "Selecionar o VLM e implementar/testar a inferência multimodal antes da integração."
    )
