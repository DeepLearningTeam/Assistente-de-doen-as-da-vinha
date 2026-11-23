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
    "classe_resposta", "imagem_sha256",
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
    tipo = manifesto["parametros"]["tipo"]
    if tipo not in {"texto", "resposta_imagem"}:
        raise ErroProjeto("A anotação de afirmações exige uma execução de respostas.")
    linhas = []
    for item in itens:
        if item["estado"] != "resposta":
            continue
        ref = item.get("referencia") or {
            "id": item["imagem_sha256"], "pergunta": item["pergunta"], "resposta_ref": "",
        }
        for numero, parte in enumerate(item["partes"], 1):
            linhas.append({
                "id_afirmacao": f"{item['chave']}:{numero}", "chave_resposta": item["chave"],
                "id_pergunta": ref["id"], "condicao": item["modo"], "temperatura": item.get("temperatura", 0),
                "repeticao": item.get("repeticao", 1), "pergunta": ref["pergunta"], "afirmacao": parte["texto"],
                "fonte_citada": " | ".join(parte["fontes"]), "trechos_no_json": " | ".join(parte["trechos"]),
                "resposta_ref": ref["resposta_ref"], "imagem_sha256": item.get("imagem_sha256", ""),
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


def posicao_fonte(passagens, doc_id, paginas=None, tolerancia=0):
    """Posição e score da primeira correspondência de documento e, opcionalmente, página."""
    for posicao, passagem in enumerate(passagens, 1):
        if passagem.get("doc", passagem.get("doc_id")) != doc_id:
            continue
        if paginas is None or any(abs(passagem["pagina"] - pagina) <= tolerancia for pagina in paginas):
            return posicao, passagem.get("score", passagem.get("similaridade"))
    return None, None


def atingiu_pagina(item):
    ref = item["referencia"]
    passagens = item.get("recuperadas", [])[:item["top_k"]]
    return posicao_fonte(passagens, ref["doc_id"], {int(ref["pagina_pdf"])})[0] is not None


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
    motivos = sorted({i.get("motivo") or "nao_registado" for itens in grupos.values()
                      for i in itens if i["estado"] == "recusa"})
    for identificador, itens in grupos.items():
        for acao, nome in (("recusar", "recusa_correta"), ("responder", "recusa_indevida")):
            relevantes = [i for i in itens if i["referencia"]["expected_action"] == acao]
            if relevantes:
                por_metrica[nome].append((sum(i["estado"] == "recusa" for i in relevantes), len(relevantes)))
                por_tipo = agrupar(relevantes, lambda i: i["referencia"].get("tipo") or "nao_registado")
                for tipo, subgrupo in por_tipo.items():
                    por_metrica[f"{nome}_tipo_{tipo}"].append((
                        sum(i["estado"] == "recusa" for i in subgrupo), len(subgrupo),
                    ))
                for motivo in motivos:
                    por_metrica[f"{nome}_motivo_{motivo}"].append((
                        sum(i["estado"] == "recusa" and (i.get("motivo") or "nao_registado") == motivo
                            for i in relevantes), len(relevantes),
                    ))
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
        classes = {}
        for chave, linhas in agrupar(rotulos, lambda r: r["chave_resposta"]).items():
            nomes = {r.get("classe_resposta", "").strip().casefold() for r in linhas}
            nomes.discard("")
            if len(nomes) > 1:
                raise ErroProjeto(f"{chave}: classe_resposta contraditória na folha de consenso.")
            if nomes:
                classes[chave] = nomes.pop()
        if rotulos:
            por_metrica["cobertura_anotacao_classe"].append((
                sum(i["chave"] in classes for i in itens), len(itens),
            ))
        pares_classe = [
            (a, b) for a, b in pares
            if all(classes.get(i["chave"]) not in {None, "sem_classe", "indeterminada"} for i in (a, b))
        ]
        if pares_classe:
            por_metrica["consistencia_classe"].append((
                sum(classes[a["chave"]] == classes[b["chave"]] for a, b in pares_classe), len(pares_classe),
            ))
            por_metrica["consistencia_classe_acao_fontes"].append((
                sum(classes[a["chave"]] == classes[b["chave"]] and decisao(a) == decisao(b)
                    for a, b in pares_classe), len(pares_classe),
            ))
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


def calibracao_imagem(itens):
    """Dez intervalos fixos; a mesma população de previsões aceites usada no ECE."""
    pares = [
        (i["visual"]["confianca"], previsao(i) == i["classe_ref"])
        for i in itens if previsao(i) and i["visual"]["confianca"] is not None
    ]
    grupos = agrupar(pares, lambda par: min(9, int(par[0] * 10)))
    return [{
        "bin": n, "limite_inf": n / 10, "limite_sup": (n + 1) / 10,
        "n": len(grupos[n]),
        "confianca_media": mean(p[0] for p in grupos[n]) if grupos[n] else None,
        "acuracia": mean(p[1] for p in grupos[n]) if grupos[n] else None,
    } for n in range(10)]


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
    bins = calibracao_imagem(itens)
    n_confianca = sum(b["n"] for b in bins)
    ece = (sum(b["n"] * abs(b["confianca_media"] - b["acuracia"]) for b in bins if b["n"])
           / n_confianca) if n_confianca else None
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
    valores = {"estado": item["estado"], "latencia_s": item["latencia_s"], "motivo": item.get("motivo", "")}
    if texto:
        valores["recusou"] = int(item["estado"] == "recusa")
        if item["rag"] and ref["expected_action"] == "responder":
            valores[f"hit_pagina@{item['top_k']}"] = int(atingiu_pagina(item))
    else:
        valores["classe_prevista"] = previsao(item)
        valores["acertou"] = int(previsao(item) == item["classe_ref"])
        valores["confianca"] = (item.get("visual") or {}).get("confianca")
    return [{**base, "metrica": chave, "valor": valor} for chave, valor in valores.items()]


def tabela_respostas_imagem(manifesto, itens, pasta_anotacoes):
    """Qualidade anotada das respostas; não calcula acurácia de classificação."""
    consenso = csv_ler(pasta_anotacoes / manifesto["execucao"] / "consenso.csv")
    tabela = []
    for modo, grupo in agrupar(itens, lambda i: i["modo"]).items():
        chaves = {i["chave"] for i in grupo}
        anotacoes = [r for r in consenso if r["chave_resposta"] in chaves]
        base = {
            "execucao": manifesto["execucao"], "split": "fotografia_sem_split_declarado", "condicao": modo,
            "modelo": manifesto["parametros"]["configuracoes"][modo]["vlm_modelo" if modo.startswith("vlm") else "llm_modelo"],
        }
        tabela.append(linha_tabela(base, "latencia_media_s", mean(i["latencia_s"] for i in grupo),
                                   len(grupo), "resposta"))
        tabela.append(linha_tabela(base, "proporcao_recusas", sum(i["estado"] == "recusa" for i in grupo) / len(grupo),
                                   len(grupo), "resposta"))
        suportes = [r["suportada"] for r in anotacoes if r["suportada"] in VALIDAS]
        if suportes:
            tabela.append(linha_tabela(base, "fidelidade", suportes.count("S") / len(suportes),
                                       len(suportes), "afirmacao"))
        por_resposta = agrupar(anotacoes, lambda r: r["chave_resposta"])
        corretas = []
        for chave, linhas in por_resposta.items():
            decisoes = {r["resposta_correta"] for r in linhas if r["resposta_correta"] in {"S", "N"}}
            if len(decisoes) > 1:
                raise ErroProjeto(f"{chave}: resposta_correta contraditória no consenso.")
            corretas.extend(decisoes)
        if corretas:
            tabela.append(linha_tabela(base, "correcao_humana", corretas.count("S") / len(corretas),
                                       len(corretas), "resposta"))
    return tabela


def resumir(nomes, destino, bootstrap):
    resultados, texto, imagem, respostas_imagem = [], [], [], []
    for nome in nomes:
        manifesto, itens = carregar(nome)
        if manifesto["parametros"]["tipo"] == "resposta_imagem":
            respostas_imagem.extend(tabela_respostas_imagem(manifesto, itens, PAPER / "fidelidade"))
            continue
        for item in itens:
            resultados.extend(resultados_item(manifesto, item))
        if manifesto["parametros"]["tipo"] == "texto":
            texto.extend(tabela_texto(manifesto, itens, bootstrap, PAPER / "fidelidade"))
        else:
            imagem.extend(tabela_imagem(manifesto, itens, bootstrap))
    if resultados:
        csv_gravar(destino / "resultados.csv", COLUNAS_RESULTADOS, resultados)
    if imagem:
        csv_gravar(destino / "tabelas/tabela1.csv", COLUNAS_TABELA, imagem)
    if texto:
        csv_gravar(destino / "tabelas/tabela2.csv", COLUNAS_TABELA, texto)
    if respostas_imagem:
        csv_gravar(destino / "tabelas/respostas_imagem.csv", COLUNAS_TABELA, respostas_imagem)
    print(f"Resultados em {destino}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="acao", required=True)
    anotar = sub.add_parser("anotar", help="Criar as folhas dos dois anotadores e do consenso.")
    anotar.add_argument("--execucao", required=True)
    tabelas = sub.add_parser("resumir", help="Gerar os resultados e as tabelas 1 e 2.")
    tabelas.add_argument("--execucoes", nargs="+", required=True)
    tabelas.add_argument("--destino", type=Path,
                         help="Pasta de exportação; por omissão, runs/<execucao>/metricas para cada execução.")
    tabelas.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    try:
        if args.acao == "anotar":
            preparar_anotacao(args.execucao)
        else:
            if args.bootstrap < 1:
                parser.error("--bootstrap tem de ser positivo.")
            if args.destino is not None:
                resumir(args.execucoes, args.destino, args.bootstrap)
            else:
                for nome in args.execucoes:
                    resumir([nome], caminho_execucao(nome) / "metricas", args.bootstrap)
    except (ErroProjeto, OSError, ValueError) as erro:
        parser.exit(1, f"{erro}\n")


if __name__ == "__main__":
    main()
