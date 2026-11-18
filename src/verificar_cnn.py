"""Ensaios reais da CNN, opcionais com Ollama/Qdrant; apenas imagens de validação."""
import argparse
import importlib.util
import platform
from importlib.metadata import version

import numpy as np

from .apoio import ErroProjeto, RAIZ, agora, gravar_json, ler_json, sha256
from .cnn_colega import _carregar_modelo, carregar_classes, carregar_modelo, preparar_imagem
from .config import MODOS, carregar_config
from .imagem import analisar_cnn, config_cnn, ler_imagem
from .indice import descrever_colecao
from .modelos import digest_modelo
from .preparar_cnn import validar_manifesto
from .rag import PERGUNTA_FOTO, responder

DESTINO = RAIZ / "verificacao/cnn"


def imagens_validacao():
    manifesto = ler_json(RAIZ / "data/splits.json")
    validar_manifesto(manifesto, [1])
    return {classe: next(i for i in manifesto["imagens"]
                        if i["classe"] == classe and i["splits"]["1"] == "validacao")
            for classe in config_cnn().classes}


def verificar_modelo():
    cfg = config_cnn()
    checkpoint = RAIZ / cfg.checkpoint
    modelo = carregar_modelo(checkpoint)
    # Código original arquivado: comparar o pré-processamento, sem o duplicar na aplicação.
    spec = importlib.util.spec_from_file_location("predict_original", RAIZ / "referencia_cnn/inference/predict.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    casos = []
    for classe, item in imagens_validacao().items():
        foto = RAIZ / "data/imagens" / item["id"]
        tensor = preparar_imagem(ler_imagem(foto))
        np.testing.assert_array_equal(tensor, original.preprocess_image(foto))
        visual = analisar_cnn(ler_imagem(foto))
        probabilidades_originais = modelo.predict(tensor, verbose=0)[0]
        np.testing.assert_allclose(list(visual.probabilidades.values()), probabilidades_originais, atol=1e-6)
        casos.append({"imagem": item["id"], "classe_ref": classe,
                      "entrada": list(tensor.shape), "dtype": str(tensor.dtype),
                      "paridade_preprocessamento_original": True,
                      "paridade_model_predict_original": True, "visual": visual.model_dump()})
    assert carregar_modelo(checkpoint) is modelo
    relatorio = {
        "data": agora(), "python": platform.python_version(), "plataforma": platform.platform(),
        "tensorflow": version("tensorflow"), "keras_distribuicao": version("keras-nightly"),
        "keras_runtime": original.keras.__version__, "checkpoint_sha256": sha256(checkpoint),
        "classes_originais": carregar_classes(), "entrada": list(modelo.input_shape),
        "saida": list(modelo.output_shape), "ativacao": modelo.layers[-1].get_config()["activation"],
        "parametros": modelo.count_params(), "camadas": [l.name for l in modelo.layers],
        "cache": _carregar_modelo.cache_info()._asdict(), "casos_validacao": casos,
        "holdout_executado": False,
    }
    gravar_json(DESTINO / "modelo_real.json", relatorio)
    print("CNN real: quatro classes de validação, pré-processamento idêntico ao original e cache confirmados.", flush=True)


def verificar_servicos():
    cfg = carregar_config()
    servicos = {"data": agora(), "ollama": {}, "qdrant": {}}
    for modelo in (cfg.llm_modelo, cfg.vlm_modelo):
        try:
            servicos["ollama"][modelo] = {"disponivel": True, "digest": digest_modelo(cfg, modelo)}
        except ErroProjeto as erro:
            servicos["ollama"][modelo] = {"disponivel": False, "mensagem": str(erro)}
    try:
        servicos["qdrant"] = {"disponivel": True, **descrever_colecao(cfg)}
    except Exception as erro:
        # Nunca guardar str/repr de uma exceção HTTP que possa conter o URL/chave.
        servicos["qdrant"] = {"disponivel": False, "tipo_erro": type(erro).__name__,
                               "mensagem": "Não foi possível consultar o esquema/payloads; verificar rede e credenciais em privado."}
    gravar_json(DESTINO / "servicos.json", servicos)
    foto = RAIZ / "data/imagens" / imagens_validacao()["esca"]["id"]
    resultados = []
    for modo in MODOS:
        config = carregar_config(modo)
        if not servicos["ollama"][config.modelo]["disponivel"] or (config.rag and not servicos["qdrant"]["disponivel"]):
            resultados.append({"modo": modo, "estado": "nao_executado", "motivo": "servico_indisponivel"})
            continue
        for com_foto in (True, False):
            chave = f"{modo}_{'foto' if com_foto else 'texto'}"
            try:
                r = responder(PERGUNTA_FOTO if com_foto else "Que é a esca da videira?", modo,
                              imagem=ler_imagem(foto) if com_foto else None, config=config, registar=True)
                gravar_json(DESTINO / f"{chave}.json", {"imagem": foto.relative_to(RAIZ).as_posix() if com_foto else None,
                                                      "resultado": r.model_dump()})
                resultados.append({"modo": modo, "com_foto": com_foto, "estado": r.estado,
                                   "motivo": r.motivo, "motor": r.visual.motor if r.visual else None,
                                   "consulta": r.consulta, "passagens": len(r.recuperadas),
                                   "fontes": [f"{f['doc']}:{f['pagina']}" for f in r.fontes]})
                print(f"{chave}: {r.estado}; recuperadas={len(r.recuperadas)}; motivo={r.motivo or '-'}", flush=True)
            except ErroProjeto as erro:
                resultados.append({"modo": modo, "com_foto": com_foto, "estado": "erro", "mensagem": str(erro)})
                print(f"{chave}: falhou; consulta o relatório sem credenciais.", flush=True)
            gravar_json(DESTINO / "percursos_reais.json", resultados)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servicos", action="store_true", help="Geração real nos quatro modos e Qdrant apenas por leitura.")
    args = parser.parse_args()
    try:
        verificar_modelo()
        if args.servicos:
            verificar_servicos()
    except (ErroProjeto, OSError, ValueError, AssertionError) as erro:
        parser.exit(1, f"Falhou a verificação da CNN: {erro}\n")


if __name__ == "__main__":
    main()
