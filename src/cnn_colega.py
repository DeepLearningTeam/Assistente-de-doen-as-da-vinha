"""Ponto de integração para uma CNN entregue com código Python de inferência."""


def prever(imagem, checkpoint):
    """Substitui este corpo pelo código do teu colega.

    Recebe uma imagem PIL em RGB e um pathlib.Path para os pesos.
    Devolve {"probabilidades": {"saudavel": p0, "esca": p1,
                                 "leaf_blight": p2, "black_rot": p3}}.
    Pode devolver também e_folha (bool), descricao (str) e mapa_atencao (imagem PIL).
    """
    raise NotImplementedError("A inferência da CNN do grupo será ligada aqui.")
