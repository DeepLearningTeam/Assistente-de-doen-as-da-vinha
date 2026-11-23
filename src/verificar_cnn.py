"""Ensaios reais da CNN, opcionais com Ollama/Qdrant; apenas imagens de validação."""
import argparse
import platform
from importlib.metadata import version

import numpy as np

from .apoio import ErroProjeto, RAIZ, agora, gravar_json, ler_json, sha256, nova_verificacao
from .cnn_resnet import LIMIAR_CONFIANCA, _carregar_modelo, carregar_classes, carregar_modelo, preparar_imagem
from .config import MODOS, carregar_config
from .imagem import analisar_cnn, config_cnn, ler_imagem
from .indice import descrever_colecao
from .modelos import digest_modelo
from .preparar_cnn import validar_manifesto
from .rag import PERGUNTA_FOTO, responder

def imagens_validacao():
    manifesto = ler_json(RAIZ / "data/splits.json")
    validar_manifesto(manifesto, [1])
    return {classe: next(i for i in manifesto["imagens"]
                        if i["classe"] == classe and i["splits"]["1"] == "validacao")
            for classe in config_cnn().classes}


def verificar_modelo(destino, apenas_modelo=False):
    cfg = config_cnn()
    checkpoint = RAIZ / cfg.checkpoint
    modelo = carregar_modelo(checkpoint)
    casos = []
    if not apenas_modelo:
        for classe, item in imagens_validacao().items():
            foto = RAIZ / "data/imagens" / item["id"]
            imagem = ler_imagem(foto)
            tensor = preparar_imagem(imagem)
            assert tensor.shape == (1, 224, 224, 3) and tensor.dtype == np.float32
            visual = analisar_cnn(imagem)
            probabilidades = modelo.predict(tensor, verbose=0)[0]
            np.testing.assert_allclose(list(visual.probabilidades.values()), probabilidades, atol=1e-6)
            # Conferir a integração da saída e da rejeição, sem executar código de treino arquivado.
            indice = int(np.argmax(probabilidades))
            aceite = cfg.classes[indice] != "outro" and float(probabilidades[indice]) >= LIMIAR_CONFIANCA
            assert (visual.estado == "ok") == aceite
            casos.append({"imagem": item["id"], "classe_ref": classe,
                          "entrada": list(tensor.shape), "dtype": str(tensor.dtype),
                          "probabilidades_consistentes": True,
                          "rejeicao_consistente": True, "visual": visual.model_dump()})
    assert carregar_modelo(checkpoint) is modelo
    relatorio = {
        "data": agora(), "python": platform.python_version(), "plataforma": platform.platform(),
        "tensorflow": version("tensorflow"), "keras_distribuicao": version("keras-nightly"),
        "checkpoint_sha256": sha256(checkpoint),
        "classes_originais": carregar_classes(), "entrada": list(modelo.input_shape),
        "saida": list(modelo.output_shape), "ativacao": modelo.layers[-1].get_config()["activation"],
        "parametros": modelo.count_params(), "camadas": [l.name for l in modelo.layers],
        "cache": _carregar_modelo.cache_info()._asdict(), "casos_validacao": casos,
        "holdout_executado": False, "inferencia_em_imagens": bool(casos),
    }
    gravar_json(destino / "modelo_real.json", relatorio)
    print("CNN: carregamento, arquitetura e classes verificados. "
          + (f"Inferência verificada em {len(casos)} imagens de validação." if casos else "Sem inferência em imagens."), flush=True)


def verificar_servicos(destino):
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
    gravar_json(destino / "servicos.json", servicos)
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
                gravar_json(destino / f"{chave}.json", {"imagem": foto.relative_to(RAIZ).as_posix() if com_foto else None,
                                                      "resultado": r.model_dump()})
                resultados.append({"modo": modo, "com_foto": com_foto, "estado": r.estado,
                                   "motivo": r.motivo, "motor": r.visual.motor if r.visual else None,
                                   "consulta": r.consulta, "passagens": len(r.recuperadas),
                                   "fontes": [f"{f['doc']}:{f['pagina']}" for f in r.fontes]})
                print(f"{chave}: {r.estado}; recuperadas={len(r.recuperadas)}; motivo={r.motivo or '-'}", flush=True)
            except ErroProjeto as erro:
                resultados.append({"modo": modo, "com_foto": com_foto, "estado": "erro", "mensagem": str(erro)})
                print(f"{chave}: falhou; consultar o relatório sem credenciais.", flush=True)
            gravar_json(destino / "percursos_reais.json", resultados)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--servicos", action="store_true", help="Geração real nos quatro modos; exige imagens de validação.")
    grupo.add_argument("--apenas-modelo", action="store_true", help="Verificar os pesos e a arquitetura sem exigir imagens.")
    args = parser.parse_args()
    destino = nova_verificacao("cnn")
    print(f"Relatórios em {destino}.", flush=True)
    try:
        verificar_modelo(destino, apenas_modelo=args.apenas_modelo)
        if args.servicos:
            verificar_servicos(destino)
    except (ErroProjeto, OSError, ValueError, AssertionError) as erro:
        gravar_json(destino / "erro.json", {"tipo_erro": type(erro).__name__, "mensagem": str(erro)})
        parser.exit(1, f"Falhou a verificação da CNN: {erro}\n")


if __name__ == "__main__":
    main()
