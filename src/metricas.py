"""Preparar anotação humana e produzir os CSV de resultados a partir dos JSON."""
import argparse
import csv
import io
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from .apoio import ErroProjeto, RAIZ, escrever_atomico, ler_json
from .avaliar import caminho_execucao
from .imagem import CLASSES

PAPER = RAIZ / "paper"
COLUNAS_ANOTACAO = (
    "id_afirmacao", "chave_resposta", "id_pergunta", "condicao", "temperatura", "repeticao",
    "pergunta", "afirmacao", "fonte_citada", "trechos_no_json", "resposta_ref",
    "suportada", "correta_face_referencia", "resposta_correta", "tipo_erro", "anotador",
)
COLUNAS_RESULTADOS = (
    "execucao", "split", "via", "modelo", "config", "seed", "repeticao",
    "id_pergunta_ou_imagem", "temperatura", "metrica", "valor", "data", "commit",
)
COLUNAS_TABELA = (
    "execucao", "split", "condicao", "modelo", "seed", "temperatura", "metrica",
    "valor", "ic95_inf", "ic95_sup", "n_unidades", "unidade",
)
VALIDAS = {"S", "N", "parcial"}


def csv_gravar(caminho, colunas, linhas):
    buffer = io.StringIO(newline="")
    escritor = csv.DictWriter(buffer, fieldnames=colunas)
    escritor.writeheader()
    escritor.writerows(linhas)
    escrever_atomico(caminho, lambda f: f.write(buffer.getvalue().encode("utf-8-sig")))


def csv_ler(caminho):
    if not caminho.is_file():
        return []
    with caminho.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def carregar(nome):
    pasta = caminho_execucao(nome)
    if not (pasta / "execucao.json").is_file():
        raise ErroProjeto(f"Não existe a execução {nome}.")
    manifesto = ler_json(pasta / "execucao.json")
    itens = [ler_json(p) for p in sorted((pasta / "itens").glob("*.json"))]
    if not itens:
        raise ErroProjeto(f"A execução {nome} não tem itens guardados.")
    esperado = manifesto["parametros"]["itens_previstos"]
    if len(itens) < esperado:
        print(f"{nome}: execução parcial ({len(itens)}/{esperado} itens).")
    return manifesto, itens


def preparar_anotacao(nome, destino=PAPER / "fidelidade"):
    manifesto, itens = carregar(nome)
    if manifesto["parametros"]["tipo"] != "texto":
        raise ErroProjeto("A anotação de afirmações exige uma execução de texto.")
    linhas = []
    for item in itens:
        if item["estado"] != "resposta":
            continue
        ref = item["referencia"]
        for numero, parte in enumerate(item["partes"], 1):
            linhas.append({
                "id_afirmacao": f"{item['chave']}:{numero}", "chave_resposta": item["chave"],
                "id_pergunta": ref["id"], "condicao": item["modo"], "temperatura": item["temperatura"],
                "repeticao": item["repeticao"], "pergunta": ref["pergunta"], "afirmacao": parte["texto"],
                "fonte_citada": " | ".join(parte["fontes"]), "trechos_no_json": " | ".join(parte["trechos"]),
                "resposta_ref": ref["resposta_ref"],
            })
    pasta = destino / nome
    for nome_csv in ("anotador_1.csv", "anotador_2.csv", "consenso.csv"):
        caminho = pasta / nome_csv
        if caminho.exists():
            print(f"Já existe: {caminho}")
        else:
            csv_gravar(caminho, COLUNAS_ANOTACAO, linhas)
    print(f"{len(linhas)} afirmações a rever em {pasta}.")


def taxa(grupos):
    numerador = sum(v[0] for v in grupos)
    denominador = sum(v[1] for v in grupos)
    return numerador / denominador if denominador else None


def agrupar(linhas, chave):
    grupos = defaultdict(list)
    for linha in linhas:
        grupos[chave(linha)].append(linha)
    return grupos


def intervalo(amostra, calcular, repeticoes):
    """Reamostra unidades completas para conservar a dependência entre repetições."""
    n = len(amostra)
    if n < 2:
        return None, None
    rng = random.Random(42)
    valores = []
    for _ in range(repeticoes):
        valor = calcular(rng.choices(amostra, k=n))
        if valor is not None:
            valores.append(valor)
    if not valores:
        return None, None
    return tuple(map(float, np.percentile(valores, [2.5, 97.5])))


def linha_tabela(base, metrica, valor, n, unidade, limites=(None, None)):
    return {
        **base, "metrica": metrica, "valor": valor, "n_unidades": n, "unidade": unidade,
        "ic95_inf": limites[0], "ic95_sup": limites[1],
    }


def atingiu_pagina(item):
    ref = item["referencia"]
    return any(p["doc"] == ref["doc_id"] and p["pagina"] == int(ref["pagina_pdf"])
               for p in item.get("recuperadas", []))


def kappa(a, b):
    n = len(a)
    if n < 2:
        return None
    a_freq, b_freq = Counter(a), Counter(b)
    observado = sum(x == y for x, y in zip(a, b)) / n
    esperado = sum(a_freq[v] * b_freq[v] for v in VALIDAS) / n**2
    return (observado - esperado) / (1 - esperado) if esperado < 1 else None


def pontuacoes_texto(grupos, anotacoes, k):
    por_metrica = defaultdict(list)
    fidelidade = {}
    for identificador, itens in grupos.items():
        for acao, nome in (("recusar", "recusa_correta"), ("responder", "recusa_indevida")):
            relevantes = [i for i in itens if i["referencia"]["expected_action"] == acao]
            if relevantes:
                por_metrica[nome].append((sum(i["estado"] == "recusa" for i in relevantes), len(relevantes)))
        com_fonte = [i for i in itens if i["rag"] and i["referencia"]["expected_action"] == "responder"]
        if com_fonte:
            por_metrica[f"recall_pagina@{k}"].append((sum(map(atingiu_pagina, com_fonte)), len(com_fonte)))
        por_metrica["latencia_media_s"].append((mean(i["latencia_s"] for i in itens), 1))
        por_metrica["falha_formato"].append((sum(i["estado"] == "erro_formato" for i in itens), len(itens)))
        por_metrica["fora_tema"].append((sum(i["estado"] == "fora_tema" for i in itens), len(itens)))
        pares = list(combinations([i for i in itens if i["estado"] != "erro_formato"], 2))
        if pares:
            def decisao(item):
                fontes = sorted((f["doc"], f["pagina"]) for f in item["fontes"])
                return item["estado"], fontes
            por_metrica["consistencia_acao_fontes"].append((
                sum(decisao(a) == decisao(b) for a, b in pares), len(pares),
            ))
        rotulos = anotacoes.get(identificador, [])
        suportes = [r["suportada"] for r in rotulos if r["suportada"] in VALIDAS]
        if suportes:
            par = (suportes.count("S"), len(suportes))
            por_metrica["fidelidade"].append(par)
            fidelidade[identificador] = par
        corretas = [r["resposta_correta"] for r in rotulos if r["resposta_correta"] in {"S", "N"}]
        if corretas:
            por_metrica["correcao_humana"].append((corretas.count("S"), len(corretas)))
    return por_metrica, fidelidade


def tabela_texto(manifesto, itens, bootstrap, pasta_anotacoes):
    nome = manifesto["execucao"]
    consenso = csv_ler(pasta_anotacoes / nome / "consenso.csv")
    pares_de_anotadores = [csv_ler(pasta_anotacoes / nome / f"anotador_{n}.csv") for n in (1, 2)]
    tabela = []
    fidelidades = {}
    for (modo, temp), condicao in agrupar(itens, lambda i: (i["modo"], i["temperatura"])).items():
        base = {
            "execucao": nome, "split": condicao[0]["split"], "condicao": modo,
            "modelo": condicao[0]["modelo"], "temperatura": temp,
        }
        anotacoes = agrupar(
            [r for r in consenso if r["condicao"] == modo and float(r["temperatura"]) == temp],
            lambda r: r["id_pergunta"],
        )
        perguntas = agrupar(condicao, lambda i: i["referencia"]["id"])
        metricas, fidelidade = pontuacoes_texto(perguntas, anotacoes, condicao[0]["top_k"])
        fidelidades[(modo, temp)] = fidelidade
        for metrica, amostra in metricas.items():
            tabela.append(linha_tabela(base, metrica, taxa(amostra), len(amostra), "pergunta",
                                       intervalo(amostra, taxa, bootstrap)))
        a, b = [
            {r["id_afirmacao"]: r["suportada"] for r in dados
             if r["condicao"] == modo and float(r["temperatura"]) == temp and r["suportada"] in VALIDAS}
            for dados in pares_de_anotadores
        ]
        ids = sorted(a.keys() & b.keys())
        if ids:
            tabela.append(linha_tabela(base, "kappa_cohen", kappa([a[i] for i in ids], [b[i] for i in ids]),
                                       len(ids), "afirmacao"))
    for temp in manifesto["parametros"]["temperaturas"]:
        rag = fidelidades.get(("llm_cnn_rag", temp), {})
        baseline = fidelidades.get(("llm_cnn", temp), {})
        pares = [(rag[i], baseline[i]) for i in sorted(rag.keys() & baseline.keys())]
        if pares:
            def diferenca(amostra):
                a, b = zip(*amostra)
                return taxa(a) - taxa(b)
            base = {
                "execucao": nome, "split": manifesto["parametros"]["split"],
                "condicao": "llm_cnn_rag - llm_cnn", "temperatura": temp,
            }
            tabela.append(linha_tabela(base, "diferenca_fidelidade_emparelhada", diferenca(pares),
                                       len(pares), "pergunta", intervalo(pares, diferenca, bootstrap)))
    return tabela


def previsao(item):
    visual = item.get("visual") or {}
    return visual.get("classe") if item["estado"] == "ok" else None


def valores_imagem(itens):
    """Uma abstenção conta como falha para a classe real; a cobertura é publicada."""
    confusao = Counter((i["classe_ref"], previsao(i)) for i in itens)
    f1, recall = {}, {}
    for classe in CLASSES:
        verdadeiros = confusao[classe, classe]
        reais = sum(n for (real, _), n in confusao.items() if real == classe)
        previstos = sum(n for (_, prev), n in confusao.items() if prev == classe)
        if reais:
            f1[classe] = 2 * verdadeiros / (reais + previstos)
            recall[f"recall_{classe}"] = verdadeiros / reais
    com_confianca = [
        (i["visual"]["confianca"], previsao(i) == i["classe_ref"])
        for i in itens if previsao(i) and i["visual"]["confianca"] is not None
    ]
    bins = agrupar(com_confianca, lambda par: min(9, int(par[0] * 10)))
    ece = None
    if com_confianca:
        ece = sum(
            len(grupo) * abs(mean(p[0] for p in grupo) - mean(p[1] for p in grupo))
            for grupo in bins.values()
        ) / len(com_confianca)
    return {
        "macro_f1": mean(f1.values()), "acuracia": sum(confusao[c, c] for c in CLASSES) / len(itens),
        "cobertura": sum(n for (_, prev), n in confusao.items() if prev is not None) / len(itens),
        "latencia_media_s": mean(i["latencia_s"] for i in itens), "ece_aceites": ece, **recall,
    }


def tabela_imagem(manifesto, itens, bootstrap):
    tabela = []
    for motor, condicao in agrupar(itens, lambda i: i["modo"]).items():
        por_seed = agrupar(condicao, lambda i: i["seed"])
        base = {"execucao": manifesto["execucao"], "split": condicao[0]["split"],
                "condicao": motor, "modelo": condicao[0]["modelo"] if len(por_seed) == 1 or motor == "vlm" else "CNN por seed"}
        for seed, grupo in [*por_seed.items(), ("todas", condicao)]:
            if seed == "todas" and len(por_seed) == 1:
                continue
            imagens = list(agrupar(grupo, lambda i: i["imagem"]).values())
            base_seed = {**base, "seed": seed, "modelo": grupo[0]["modelo"] if seed != "todas" else base["modelo"]}
            metricas = valores_imagem(grupo)
            # Calcula todas as métricas na mesma reamostragem de imagens.
            rng = random.Random(42)
            valores_boot = defaultdict(list)
            if len(imagens) > 1:
                for _ in range(bootstrap):
                    amostra = [i for subconjunto in rng.choices(imagens, k=len(imagens)) for i in subconjunto]
                    for metrica, valor in valores_imagem(amostra).items():
                        if valor is not None:
                            valores_boot[metrica].append(valor)
            for metrica, valor in metricas.items():
                valores = valores_boot.get(metrica)
                limites = tuple(map(float, np.percentile(valores, [2.5, 97.5]))) if valores else (None, None)
                tabela.append(linha_tabela(base_seed, metrica, valor, len(imagens), "imagem", limites))
        if len(por_seed) > 1:
            f1_seeds = [valores_imagem(grupo)["macro_f1"] for grupo in por_seed.values()]
            tabela.append(linha_tabela(base, "macro_f1_media_seeds", mean(f1_seeds), len(f1_seeds), "seed"))
            tabela.append(linha_tabela(base, "macro_f1_dp_seeds", stdev(f1_seeds), len(f1_seeds), "seed"))
    return tabela


def resultados_item(manifesto, item):
    texto = item["tipo"] == "texto"
    ref = item["referencia"] if texto else {}
    base = {
        "execucao": manifesto["execucao"], "split": item["split"], "via": item["modo"],
        "modelo": item["modelo"], "config": f"{item['modo']}.yaml",
        "seed": item["seed"], "repeticao": item.get("repeticao"),
        "id_pergunta_ou_imagem": ref["id"] if texto else item["imagem"],
        "temperatura": item.get("temperatura"), "data": item["data"],
        "commit": manifesto["parametros"]["commit"],
    }
    valores = {"estado": item["estado"], "latencia_s": item["latencia_s"]}
    if texto:
        valores["recusou"] = int(item["estado"] == "recusa")
        if item["rag"] and ref["expected_action"] == "responder":
            valores[f"hit_pagina@{item['top_k']}"] = int(atingiu_pagina(item))
    else:
        valores["classe_prevista"] = previsao(item)
        valores["acertou"] = int(previsao(item) == item["classe_ref"])
        valores["confianca"] = (item.get("visual") or {}).get("confianca")
    return [{**base, "metrica": chave, "valor": valor} for chave, valor in valores.items()]


def resumir(nomes, destino, bootstrap):
    resultados, texto, imagem = [], [], []
    for nome in nomes:
        manifesto, itens = carregar(nome)
        if manifesto["parametros"]["tipo"] == "resposta_imagem":
            raise ErroProjeto("A comparação de respostas com fotografia exige revisão humana dos JSON; não é classificação.")
        for item in itens:
            resultados.extend(resultados_item(manifesto, item))
        if manifesto["parametros"]["tipo"] == "texto":
            texto.extend(tabela_texto(manifesto, itens, bootstrap, PAPER / "fidelidade"))
        else:
            imagem.extend(tabela_imagem(manifesto, itens, bootstrap))
    csv_gravar(destino / "resultados.csv", COLUNAS_RESULTADOS, resultados)
    if imagem:
        csv_gravar(destino / "tabelas/tabela1.csv", COLUNAS_TABELA, imagem)
    if texto:
        csv_gravar(destino / "tabelas/tabela2.csv", COLUNAS_TABELA, texto)
    print(f"Resultados em {destino}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="acao", required=True)
    anotar = sub.add_parser("anotar", help="Criar as folhas dos dois anotadores e do consenso.")
    anotar.add_argument("--execucao", required=True)
    tabelas = sub.add_parser("resumir", help="Gerar os resultados e as tabelas 1 e 2.")
    tabelas.add_argument("--execucoes", nargs="+", required=True)
    tabelas.add_argument("--destino", type=Path, default=PAPER)
    tabelas.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    try:
        if args.acao == "anotar":
            preparar_anotacao(args.execucao)
        else:
            if args.bootstrap < 1:
                parser.error("--bootstrap tem de ser positivo.")
            resumir(args.execucoes, args.destino, args.bootstrap)
    except (ErroProjeto, OSError, ValueError) as erro:
        parser.exit(1, f"{erro}\n")


if __name__ == "__main__":
    main()
