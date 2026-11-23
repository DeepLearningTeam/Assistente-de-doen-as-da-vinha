"""Regressões da avaliação; sem pesos nem serviços remotos."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from src import analise, apoio, avaliar, avaliar_recuperacao, indice, metricas, rag
from src.apoio import ErroProjeto, sha256
from src.config import carregar_config
from src.imagem import Visual
from src.preparar_cnn import validar_manifesto


def item(chave="r1", **alteracoes):
    return {
        "chave": chave, "estado": "resposta", "fontes": [], "rag": False,
        "referencia": {"expected_action": "responder", "doc_id": "D02", "pagina_pdf": "6"},
        "latencia_s": 1, "top_k": 5, **alteracoes,
    }


class MetricasAvaliacao(unittest.TestCase):
    def test_correspondencia_de_fontes_usa_id_e_pagina(self):
        passagens = [
            {"doc_id": "D01", "pagina": 6, "score": 0.9},
            {"doc_id": "D02", "pagina": 5, "score": 0.8},
            {"doc_id": "D02", "pagina": 6, "score": 0.7},
        ]
        self.assertEqual(metricas.posicao_fonte(passagens, "D02"), (2, 0.8))
        self.assertEqual(metricas.posicao_fonte(passagens, "D02", {6}), (3, 0.7))
        self.assertEqual(metricas.posicao_fonte(passagens, "D02", {6}, tolerancia=1), (2, 0.8))
        self.assertEqual(metricas.posicao_fonte(passagens, "D03", {6}), (None, None))
        self.assertFalse(metricas.atingiu_pagina(item(recuperadas=passagens, top_k=2)))
        self.assertTrue(metricas.atingiu_pagina(item(recuperadas=passagens, top_k=3)))

    def test_calibracao_inclui_confianca_um_e_exclui_abstencoes(self):
        itens = [
            {"estado": "ok", "classe_ref": "esca", "visual": {"classe": "esca", "confianca": 1.0}},
            {"estado": "ok", "classe_ref": "esca", "visual": {"classe": "black_rot", "confianca": 0.8}},
            {"estado": "incerta", "classe_ref": "esca", "visual": {"classe": "esca", "confianca": 0.4}},
        ]
        bins = metricas.calibracao_imagem(itens)
        self.assertEqual(sum(b["n"] for b in bins), 2)
        self.assertEqual(bins[9]["acuracia"], 1)
        self.assertEqual(bins[8]["acuracia"], 0)
        ece = sum(b["n"] * abs(b["confianca_media"] - b["acuracia"]) for b in bins if b["n"]) / 2
        self.assertAlmostEqual(ece, 0.4)

    def test_analise_exporta_erros_confiantes_e_abstencoes(self):
        itens = [{
            "modo": "cnn", "seed": 1, "imagem": f"foto{n}.png", "chave": f"i{n}",
            "estado": estado, "classe_ref": "esca", "visual": {"classe": classe, "confianca": confianca},
        } for n, (estado, classe, confianca) in enumerate([
            ("ok", "black_rot", 0.95), ("ok", "esca", 0.8), ("rejeitada", None, None),
        ])]
        with tempfile.TemporaryDirectory() as tmp, patch.object(analise, "carregar", return_value=(
            {"parametros": {"tipo": "imagem"}}, itens,
        )):
            analise.analisar("ensaio", Path(tmp))
            casos = metricas.csv_ler(Path(tmp) / "piores_casos.csv")
            self.assertEqual([c["chave"] for c in casos], ["i0", "i2"])
            self.assertEqual(sum(int(b["n"]) for b in metricas.csv_ler(Path(tmp) / "calibracao.csv")), 2)

    def test_mesmas_fontes_nao_implicam_mesma_classe(self):
        itens = [item("r1"), item("r2")]
        # A classe visual é igual, mas as respostas humanas identificam doenças diferentes.
        for i in itens:
            i["visual"] = {"classe": "esca"}
        anotacoes = {"Q1": [
            {"chave_resposta": "r1", "classe_resposta": "esca", "suportada": "N", "resposta_correta": ""},
            {"chave_resposta": "r2", "classe_resposta": "black_rot", "suportada": "N", "resposta_correta": ""},
        ]}
        valores, _ = metricas.pontuacoes_texto({"Q1": itens}, anotacoes, 5)
        self.assertEqual(metricas.taxa(valores["consistencia_acao_fontes"]), 1)
        self.assertEqual(metricas.taxa(valores["consistencia_classe"]), 0)
        self.assertEqual(metricas.taxa(valores["consistencia_classe_acao_fontes"]), 0)
        self.assertEqual(metricas.taxa(valores["fidelidade"]), 0)

    def test_classe_nao_anotada_ou_sem_classe_nao_cria_acordo_artificial(self):
        itens = [item("r1"), item("r2")]
        valores, _ = metricas.pontuacoes_texto({"Q1": itens}, {}, 5)
        self.assertNotIn("consistencia_classe", valores)
        anotacoes = {"Q1": [
            {"chave_resposta": i["chave"], "classe_resposta": "sem_classe", "suportada": "", "resposta_correta": ""}
            for i in itens
        ]}
        valores, _ = metricas.pontuacoes_texto({"Q1": itens}, anotacoes, 5)
        self.assertNotIn("consistencia_classe", valores)
        self.assertEqual(metricas.taxa(valores["cobertura_anotacao_classe"]), 1)

    def test_classes_contraditorias_na_mesma_resposta_falham(self):
        linhas = [{"chave_resposta": "r1", "classe_resposta": c} for c in ("esca", "black_rot")]
        with self.assertRaisesRegex(ErroProjeto, "contraditória"):
            metricas.pontuacoes_texto({"Q1": [item()]}, {"Q1": linhas}, 5)

    def test_motivos_de_recusa_usam_todas_as_perguntas_no_denominador(self):
        grupos = {"Q1": [item(estado="recusa", motivo="documento_ausente")], "Q2": [item()]}
        valores, _ = metricas.pontuacoes_texto(grupos, {}, 5)
        self.assertEqual(metricas.taxa(valores["recusa_indevida"]), 0.5)
        self.assertEqual(metricas.taxa(valores["recusa_indevida_motivo_documento_ausente"]), 0.5)

    def test_recusas_por_tipo_separam_acoes_e_conservam_falhas_no_denominador(self):
        def caso(tipo, acao, estado):
            return item(estado=estado, referencia={
                "tipo": tipo, "expected_action": acao, "doc_id": "D02", "pagina_pdf": "6",
            })

        grupos = {
            "Q1": [caso("regulamentacao", "recusar", "recusa")],
            "Q2": [caso("sintomas", "recusar", estado) for estado in ("recusa", "resposta", "fora_tema")],
            "Q3": [caso("sintomas", "recusar", "erro_formato")],
            "Q4": [caso("tratamento", "recusar", "resposta")],
            "Q5": [caso("sintomas", "responder", estado) for estado in ("recusa", "resposta")],
        }
        valores, _ = metricas.pontuacoes_texto(grupos, {}, 5)
        self.assertAlmostEqual(metricas.taxa(valores["recusa_correta"]), 2 / 6)
        self.assertEqual(metricas.taxa(valores["recusa_correta_tipo_regulamentacao"]), 1)
        self.assertEqual(metricas.taxa(valores["recusa_correta_tipo_sintomas"]), 1 / 4)
        self.assertEqual(metricas.taxa(valores["recusa_correta_tipo_tratamento"]), 0)
        self.assertEqual(metricas.taxa(valores["recusa_indevida_tipo_sintomas"]), 1 / 2)

    def test_execucoes_antigas_sem_tipo_preservam_taxa_e_assinalam_tipo_ausente(self):
        valores, _ = metricas.pontuacoes_texto({"Q1": [item(estado="recusa")], "Q2": [item()]}, {}, 5)
        self.assertEqual(metricas.taxa(valores["recusa_indevida"]), 0.5)
        self.assertEqual(metricas.taxa(valores["recusa_indevida_tipo_nao_registado"]), 0.5)

    def test_hit_pagina_respeita_k(self):
        i = item(recuperadas=[{"doc": "D01", "pagina": 1}] * 5 + [{"doc": "D02", "pagina": 6}])
        self.assertFalse(metricas.atingiu_pagina(i))
        i["recuperadas"][0] = {"doc": "D02", "pagina": 6}
        self.assertTrue(metricas.atingiu_pagina(i))

    def test_anonacoes_antigas_sem_coluna_classe_continuam_validas(self):
        linhas = [{"chave_resposta": "r1", "suportada": "S", "resposta_correta": "S"}]
        valores, _ = metricas.pontuacoes_texto({"Q1": [item()]}, {"Q1": linhas}, 5)
        self.assertEqual(metricas.taxa(valores["fidelidade"]), 1)
        self.assertNotIn("consistencia_classe", valores)


class PassagemFonte(unittest.TestCase):
    def setUp(self):
        self.pergunta = {"id": "Q1", "pergunta": "Por onde entra o fungo?", "passagem_ref": "Entrada por feridas.",
                         "resposta_ref": "Entra por feridas de poda.", "expected_action": "responder",
                         "doc_id": "D02", "pagina_pdf": "6", "split": "validacao", "estado_validacao": "pendente",
                         "tipo": "biologia"}
        self.texto = "O fungo entra por feridas recentes, incluindo feridas de poda."

    def fonte(self, **alteracoes):
        return {"id_pergunta": "Q1", "pergunta": self.pergunta["pergunta"], "criterio_evidencia": "Entrada por feridas.",
                "doc_id": "D02", "pagina_pdf": "6", "passagem_id": "origem",
                "texto_passagem": self.texto, "passagem_sha256": metricas.hash_passagem(self.texto),
                "evidencia": "entra por feridas recentes, incluindo feridas de poda", "revisor": "A1",
                "estado_validacao": "validada", **alteracoes}

    def carregar(self, linhas):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "fontes.csv"
            metricas.csv_gravar(caminho, metricas.COLUNAS_FONTES, linhas)
            return metricas.carregar_fontes(caminho)

    def test_pagina_certa_com_passagem_errada_nao_acerta_passagem(self):
        fontes = self.carregar([self.fonte()])
        errada = {"doc": "D02", "pagina": 6, "id": "origem", "texto": "Esta página inclui várias doenças da madeira."}
        i = item(rag=True, referencia=self.pergunta, recuperadas=[errada])
        self.assertTrue(metricas.atingiu_pagina(i))
        self.assertEqual(metricas.posicao_passagem([errada], self.pergunta, fontes), (True, None))
        valores, _ = metricas.pontuacoes_texto({"Q1": [i]}, {}, 5, fontes)
        self.assertEqual(metricas.taxa(valores["recall_pagina@5"]), 1)
        self.assertEqual(metricas.taxa(valores["recall_passagem@5"]), 0)
        self.assertEqual(metricas.taxa(valores["cobertura_recall_passagem@5"]), 1)

    def test_equivalentes_so_contam_quando_validadas(self):
        texto = "As feridas de poda recentes permitem a entrada do fungo."
        equivalente = self.fonte(doc_id="D03", pagina_pdf="9", texto_passagem=texto,
                                 passagem_sha256=metricas.hash_passagem(texto), evidencia=texto)
        passagem = {"doc_id": "D03", "pagina": 9, "texto": texto, "id": "outro-id"}
        for estado, rank in (("validada", 1), ("pendente_validacao_humana", None), ("rejeitada", None)):
            with self.subTest(estado=estado):
                fontes = self.carregar([self.fonte(), {**equivalente, "estado_validacao": estado}])
                self.assertEqual(metricas.posicao_passagem([passagem], self.pergunta, fontes), (True, rank))

    def test_texto_com_espacos_equivalentes_e_documento_pagina_obrigatorios(self):
        fontes = self.carregar([self.fonte()])
        passagem = {"doc": "D02", "pagina": 6, "texto": self.texto.replace(" ", "\n  "), "id": "outro-id"}
        self.assertEqual(metricas.posicao_passagem([passagem], self.pergunta, fontes), (True, 1))
        for campo, valor in (("doc", "D03"), ("pagina", 7)):
            with self.subTest(campo=campo):
                self.assertEqual(metricas.posicao_passagem([{**passagem, campo: valor}], self.pergunta, fontes), (True, None))

    def test_passagem_certa_fora_das_cinco_nao_conta(self):
        fontes = self.carregar([self.fonte()])
        certa = {"doc": "D02", "pagina": 6, "texto": self.texto}
        i = item(rag=True, referencia=self.pergunta, recuperadas=[{**certa, "texto": "Outro trecho."}] * 5 + [certa])
        valores, _ = metricas.pontuacoes_texto({"Q1": [i]}, {}, 5, fontes)
        self.assertEqual(metricas.taxa(valores["recall_passagem@5"]), 0)
        i["recuperadas"][4] = certa
        valores, _ = metricas.pontuacoes_texto({"Q1": [i]}, {}, 5, fontes)
        self.assertEqual(metricas.taxa(valores["recall_passagem@5"]), 1)

    def test_pendentes_e_perguntas_alteradas_nao_sao_avaliaveis(self):
        for linhas in ([], [self.fonte(estado_validacao="pendente_validacao_humana")],
                       [self.fonte(pergunta="Uma pergunta antiga.")], [self.fonte(criterio_evidencia="Critério antigo.")]):
            with self.subTest(linhas=linhas):
                fontes = self.carregar(linhas)
                i = item(rag=True, referencia=self.pergunta, recuperadas=[])
                self.assertEqual(metricas.posicao_passagem([], self.pergunta, fontes), (None, None))
                valores, _ = metricas.pontuacoes_texto({"Q1": [i]}, {}, 5, fontes)
                self.assertNotIn("recall_passagem@5", valores)
                self.assertEqual(metricas.taxa(valores["cobertura_recall_passagem@5"]), 0)

    def test_cobertura_exclui_pendentes_do_denominador_do_recall(self):
        outra = {**self.pergunta, "id": "Q2"}
        pendente = {**self.pergunta, "id": "Q3"}
        fontes = self.carregar([self.fonte(), self.fonte(id_pergunta="Q2")])
        itens = [item("r1", rag=True, referencia=self.pergunta, recuperadas=[{"doc_id": "D02", "pagina": 6, "texto": self.texto}]),
                 item("r2", rag=True, referencia=outra, recuperadas=[]),
                 item("r3", rag=True, referencia=pendente, recuperadas=[])]
        valores, _ = metricas.pontuacoes_texto({i["referencia"]["id"]: [i] for i in itens}, {}, 5, fontes)
        self.assertEqual(metricas.taxa(valores["recall_passagem@5"]), 1 / 2)
        self.assertEqual(metricas.taxa(valores["cobertura_recall_passagem@5"]), 2 / 3)
        # Sem recuperação, esta métrica não é atribuída à baseline.
        valores, _ = metricas.pontuacoes_texto({"Q1": [item(referencia=self.pergunta)]}, {}, 5, fontes)
        self.assertNotIn("recall_passagem@5", valores)
        self.assertNotIn("cobertura_recall_passagem@5", valores)
        with patch.object(metricas, "carregar_fontes", side_effect=AssertionError("Referências chamadas na baseline")):
            metricas.pontuacoes_texto({"Q1": [item(referencia=self.pergunta)]}, {}, 5)

    def test_validacao_exige_evidencia_literal_revisor_e_conteudo_coerente(self):
        for alteracoes in ({"evidencia": ""}, {"evidencia": "Evidência ausente do texto."},
                           {"revisor": ""}, {"texto_passagem": "Texto alterado."}, {"pagina_pdf": "0"}):
            with self.subTest(alteracoes=alteracoes), self.assertRaises(ErroProjeto):
                self.carregar([self.fonte(**alteracoes)])

    def test_preparacao_le_indice_sem_modelos_e_preserva_ficheiro_existente(self):
        perguntas = [self.pergunta, {**self.pergunta, "id": "Q2", "pagina_pdf": "8"},
                     {**self.pergunta, "id": "Q3", "split": "teste"},
                     {**self.pergunta, "id": "Q4", "expected_action": "recusar"}]
        ponto = SimpleNamespace(id="id-real", payload={"doc_id": "D02", "pagina": 6, "texto": self.texto})
        with tempfile.TemporaryDirectory() as tmp, patch.object(avaliar, "ler_perguntas", return_value=perguntas), \
                patch.object(indice, "descrever_colecao", return_value={"colecao": "ensaio"}), \
                patch.object(indice, "obter_cliente") as obter, \
                patch.object(indice, "obter_modelo_bge", side_effect=AssertionError("Embeddings chamados")):
            obter.return_value.scroll.side_effect = [([ponto], 10), ([], None)]
            destino = Path(tmp) / "fontes.csv"
            metricas.preparar_fontes(destino, "validacao")
            linhas = metricas.csv_ler(destino)
            self.assertEqual({r["id_pergunta"] for r in linhas}, {"Q1", "Q2"})
            self.assertTrue(all(r["estado_validacao"] == "pendente_validacao_humana" for r in linhas))
            self.assertTrue(all(not r["evidencia"] and not r["revisor"] for r in linhas))
            self.assertEqual(linhas[0]["texto_passagem"], self.texto)
            self.assertEqual(linhas[1]["texto_passagem"], "")
            self.assertEqual(obter.return_value.scroll.call_count, 2)
            self.assertTrue(all(c.kwargs["with_vectors"] is False for c in obter.return_value.scroll.call_args_list))
            self.assertEqual(metricas.carregar_fontes(destino), {})
            anterior = destino.read_bytes()
            with self.assertRaisesRegex(ErroProjeto, "preservado"):
                metricas.preparar_fontes(destino, "validacao")
            self.assertEqual(destino.read_bytes(), anterior)

    def test_tabela_exporta_recall_vazio_com_cobertura_zero(self):
        manifesto = {"execucao": "ensaio", "parametros": {"temperaturas": [0], "split": "validacao"}}
        i = item(rag=True, referencia=self.pergunta, recuperadas=[], split="validacao", modelo="simulado",
                 modo="llm_cnn_rag", temperatura=0)
        with tempfile.TemporaryDirectory() as tmp, patch.object(metricas, "carregar_fontes", return_value={}):
            pasta = Path(tmp)
            tabela = metricas.tabela_texto(manifesto, [i], 10, pasta)
            metricas.csv_gravar(pasta / "tabela.csv", metricas.COLUNAS_TABELA, tabela)
            tabela = {r["metrica"]: r for r in metricas.csv_ler(pasta / "tabela.csv")}
        self.assertEqual(tabela["recall_passagem@5"]["valor"], "")
        self.assertEqual(tabela["recall_passagem@5"]["n_unidades"], "0")
        self.assertEqual(tabela["cobertura_recall_passagem@5"]["valor"], "0.0")
        self.assertEqual(tabela["recall_pagina@5"]["valor"], "0.0")

    def test_avaliador_recuperacao_separa_pagina_passagem_e_pendencias(self):
        for fontes, esperado in ((self.carregar([self.fonte()]), "0"), ({}, "")):
            with self.subTest(esperado=esperado), tempfile.TemporaryDirectory() as tmp:
                destino = Path(tmp) / "recall.csv"
                errada = {"doc_id": "D02", "pagina": 6, "texto": "Trecho sem a evidência.", "score": 0.9,
                          "nome_documento": "D02.pdf"}
                with patch("sys.argv", ["recuperacao", "--reranker", "--saida", str(destino)]), \
                        patch.object(avaliar_recuperacao, "carregar_perguntas", return_value=[self.pergunta]), \
                        patch.object(avaliar_recuperacao, "obter_cliente"), \
                        patch.object(avaliar_recuperacao, "descrever_colecao", return_value={"documentos": ["D02"]}), \
                        patch.object(avaliar_recuperacao, "carregar_fontes", return_value=fontes), \
                        patch.object(avaliar_recuperacao, "pesquisar_query_qdrant", return_value=[errada]) as buscar:
                    avaliar_recuperacao.main()
                linha = metricas.csv_ler(destino)[0]
                self.assertEqual(linha["rank_pág ±0"], "1")
                self.assertEqual(linha["hit_passagem"], esperado)
                self.assertEqual(buscar.call_args.args[0], self.pergunta["pergunta"])


class DoencaNaResposta(unittest.TestCase):
    def test_conta_por_resposta_e_publica_cobertura_e_exclusoes(self):
        casos = [
            ("resposta", "S", "esca", "conhecida"),
            ("resposta", "S", "black_rot", "conhecida"),
            ("resposta", "S", "sem_classe", "conhecida"),
            ("resposta", "S", "indeterminada", "conhecida"),
            ("recusa", "S", "esca", "conhecida"),
            ("imagem_rejeitada", "S", "", "conhecida"),
            ("resposta", "S", "", "conhecida"),
            ("resposta", "S", "esca", "desconhecida"),
            ("resposta", "N", "esca", "conhecida"),
            ("resposta", "", "esca", "conhecida"),
            ("erro_formato", "S", "", "conhecida"),
        ]
        itens, linhas = [], []
        for n, (estado, pedido, classe, resumo) in enumerate(casos):
            itens.append(item(str(n), estado=estado, imagem_sha256=resumo,
                              visual={"classe": "black_rot" if n == 0 else "esca"}))
            linhas.append({"chave_resposta": str(n), "pedido_identificacao": pedido, "classe_resposta": classe})
        # Duas afirmações da mesma resposta não duplicam a unidade avaliada.
        linhas.extend([dict(linhas[0]), {"chave_resposta": "0"}])
        valores = metricas.pontuacoes_doenca_resposta(itens, linhas, {"conhecida": "esca"})
        self.assertEqual(valores["acuracia_doenca_resposta"], (1 / 7, 7))
        self.assertEqual(valores["cobertura_acuracia_doenca_resposta"], (7 / 9, 9))
        self.assertEqual(valores["cobertura_pedido_identificacao"], (10 / 11, 11))
        esperadas = {"acertos": 1, "erros": 1, "recusas": 2, "sem_identificacao": 2, "falhas": 1,
                     "anotacoes_pendentes": 1, "sem_referencia": 1, "pedidos_pendentes": 1, "fora_identificacao": 1}
        for nome, n in esperadas.items():
            self.assertEqual(valores[f"doenca_resposta_n_{nome}"], (n, 11))

    def test_folhas_antigas_e_pendentes_nao_produzem_acuracia_ficticia(self):
        for linhas in ([], [{"chave_resposta": "r1", "classe_resposta": "esca"}],
                       [{"chave_resposta": "r1", "pedido_identificacao": None, "classe_resposta": None}]):
            with self.subTest(linhas=linhas):
                valores = metricas.pontuacoes_doenca_resposta(
                    [item(imagem_sha256="foto")], linhas, {"foto": "esca"},
                )
                self.assertEqual(valores["acuracia_doenca_resposta"], (None, 0))
                self.assertEqual(valores["cobertura_acuracia_doenca_resposta"], (None, 0))
                self.assertEqual(valores["cobertura_pedido_identificacao"], (0, 1))
        valores = metricas.pontuacoes_doenca_resposta([item(imagem_sha256="foto")], [
            {"chave_resposta": "r1", "pedido_identificacao": "S"},
        ], {"foto": "esca"})
        self.assertEqual(valores["acuracia_doenca_resposta"], (None, 0))
        self.assertEqual(valores["cobertura_acuracia_doenca_resposta"], (0, 1))

    def test_saudavel_acerta_e_listas_ou_outras_doencas_nao_acertam_por_inclusao(self):
        for classe, esperado in ((" Saudavel ", 1), ("esca;saudavel", 0), ("mildio", 0)):
            with self.subTest(classe=classe):
                valores = metricas.pontuacoes_doenca_resposta([item(imagem_sha256="foto")], [
                    {"chave_resposta": "r1", "pedido_identificacao": "S", "classe_resposta": classe},
                ], {"foto": "saudavel"})
                self.assertEqual(valores["acuracia_doenca_resposta"], (esperado, 1))

    def test_contradicoes_e_marcacao_de_pedido_invalida_sao_recusadas(self):
        for campo, valores in (("classe_resposta", ("esca", "black_rot")), ("pedido_identificacao", ("S", "N"))):
            with self.subTest(campo=campo), self.assertRaisesRegex(ErroProjeto, "contraditório"):
                metricas.pontuacoes_doenca_resposta([item()], [
                    {"chave_resposta": "r1", campo: v} for v in valores
                ], {})
        with self.assertRaisesRegex(ErroProjeto, "exige S, N"):
            metricas.pontuacoes_doenca_resposta([item()], [
                {"chave_resposta": "r1", "pedido_identificacao": "talvez"},
            ], {})

    def test_referencia_exige_hash_conhecido_e_preserva_as_metricas_anteriores(self):
        manifesto = {"execucao": "ensaio", "parametros": {"configuracoes": {"vlm": {"vlm_modelo": "simulado"}}}}
        itens = [item("certa", modo="vlm", imagem_sha256="hash-conhecido", visual={"classe": "black_rot"}),
                 item("externa", modo="vlm", imagem_sha256="hash-externo", imagem="Esca_0.jpg"),
                 item("sem_hash", modo="vlm", imagem_sha256=None)]
        linhas = [{"chave_resposta": i["chave"], "pedido_identificacao": "S", "classe_resposta": "esca",
                   "suportada": "S", "resposta_correta": "N"} for i in itens]
        with tempfile.TemporaryDirectory() as tmp, patch.object(metricas, "ler_json", return_value={"imagens": [
            {"sha256": "hash-conhecido", "classe": "esca"}, {"sha256": None, "classe": "esca"},
        ]}):
            pasta = Path(tmp)
            metricas.csv_gravar(pasta / "ensaio/consenso.csv", metricas.COLUNAS_ANOTACAO, linhas)
            tabela = metricas.tabela_respostas_imagem(manifesto, itens, pasta)
        valores = {r["metrica"]: r["valor"] for r in tabela}
        self.assertEqual(valores["acuracia_doenca_resposta"], 1)
        self.assertEqual(valores["cobertura_acuracia_doenca_resposta"], 1 / 3)
        self.assertEqual(valores["doenca_resposta_n_sem_referencia"], 2)
        self.assertEqual(valores["fidelidade"], 1)
        self.assertEqual(valores["correcao_humana"], 0)
        self.assertEqual(valores["latencia_media_s"], 1)
        self.assertEqual(valores["proporcao_recusas"], 0)

    def test_manifesto_com_classes_contraditorias_para_o_mesmo_hash_falha(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(metricas, "ler_json", return_value={"imagens": [
            {"sha256": "igual", "classe": "esca"}, {"sha256": "igual", "classe": "black_rot"},
        ]}), self.assertRaisesRegex(ErroProjeto, "classes contraditórias"):
            metricas.tabela_respostas_imagem({"execucao": "ensaio"}, [], Path(tmp))

    def test_exporta_acuracia_pelos_quatro_modos_com_referencia_do_manifesto(self):
        # Só metadados de validação: não executa modelos nem produz resultados científicos.
        referencia = next(i for i in apoio.ler_json(apoio.FICHEIRO_SPLITS)["imagens"]
                          if i["splits"]["1"] == "validacao" and i["sha256"])
        modos = ("llm_cnn", "llm_cnn_rag", "vlm", "vlm_rag")
        manifesto = {"execucao": "ensaio", "parametros": {
            "tipo": "resposta_imagem", "configuracoes": {m: carregar_config(m).model_dump() for m in modos},
        }}
        itens = [item(m, modo=m, imagem_sha256=referencia["sha256"]) for m in modos]
        linhas = [{"chave_resposta": m, "pedido_identificacao": "S", "classe_resposta": referencia["classe"]}
                  for m in modos]
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            metricas.csv_gravar(raiz / "fidelidade/ensaio/consenso.csv", metricas.COLUNAS_ANOTACAO, linhas)
            with patch.object(metricas, "PAPER", raiz), \
                    patch.object(metricas, "carregar", return_value=(manifesto, itens)):
                metricas.resumir(["ensaio"], raiz / "resultados", 10)
            tabela = metricas.csv_ler(raiz / "resultados/tabelas/respostas_imagem.csv")
        acuracias = [r for r in tabela if r["metrica"] == "acuracia_doenca_resposta"]
        self.assertEqual({r["condicao"] for r in acuracias}, set(modos))
        self.assertTrue(all(float(r["valor"]) == 1 and r["n_unidades"] == "1" for r in acuracias))

    def test_folhas_incluem_recusas_e_falhas_sem_inventar_anotacoes(self):
        itens = [item(str(n), estado=estado, imagem_sha256="foto", pergunta="Que doença é esta?", referencia=None,
                      modo="vlm", texto="Texto final.", partes=[]) for n, estado in enumerate(
            ("recusa", "imagem_rejeitada", "erro_formato"),
        )]
        with tempfile.TemporaryDirectory() as tmp, patch.object(metricas, "carregar", return_value=(
            {"parametros": {"tipo": "resposta_imagem"}}, itens,
        )):
            pasta = Path(tmp)
            metricas.preparar_anotacao("ensaio", pasta)
            for nome in ("anotador_1", "anotador_2", "consenso"):
                linhas = metricas.csv_ler(pasta / f"ensaio/{nome}.csv")
                self.assertEqual({r["chave_resposta"] for r in linhas}, {"0", "1", "2"})
                self.assertTrue(all(not r[c] for r in linhas for c in (
                    "pedido_identificacao", "classe_resposta", "resposta_ref", "suportada", "resposta_correta",
                )))
            caminho = pasta / "ensaio/consenso.csv"
            caminho.write_text("anotacao manual a preservar", encoding="utf-8")
            metricas.preparar_anotacao("ensaio", pasta)
            self.assertEqual(caminho.read_text(encoding="utf-8"), "anotacao manual a preservar")


class PercursosAvaliacao(unittest.TestCase):
    def test_metricas_exportam_por_execucao_sem_alterar_paper(self):
        with patch("sys.argv", ["metricas", "resumir", "--execucoes", "ensaio_a", "ensaio_b"]), \
                patch.object(metricas, "resumir") as resumir:
            metricas.main()
        self.assertEqual(resumir.call_count, 2)
        for chamada, nome in zip(resumir.call_args_list, ("ensaio_a", "ensaio_b")):
            self.assertEqual(chamada.args, ([nome], avaliar.caminho_execucao(nome) / "metricas", 1000))

    def test_exportacao_conjunta_exige_destino_explicito(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch("sys.argv", ["metricas", "resumir", "--execucoes", "ensaio_a", "ensaio_b", "--destino", tmp]), \
                patch.object(metricas, "resumir") as resumir:
            metricas.main()
            resumir.assert_called_once_with(["ensaio_a", "ensaio_b"], Path(tmp), 1000)

    def test_analise_tem_destino_local_a_execucao(self):
        with patch("sys.argv", ["analise", "--execucao", "ensaio"]), patch.object(analise, "analisar") as analisar:
            analise.main()
        analisar.assert_called_once_with("ensaio", avaliar.caminho_execucao("ensaio") / "analise", False)

    def test_verificacoes_nao_substituem_relatorios_anteriores(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(apoio, "RAIZ", Path(tmp)):
            (Path(tmp) / "PROTOCOLO.md").write_text("Protocolo de ensaio", encoding="utf-8")
            primeira = apoio.nova_verificacao("ensaio")
            (primeira / "resultado.txt").write_text("preservar", encoding="utf-8")
            segunda = apoio.nova_verificacao("ensaio")
            self.assertNotEqual(primeira, segunda)
            self.assertEqual(primeira.parent, Path(tmp) / "runs/verificacao")
            self.assertEqual((primeira / "resultado.txt").read_text(encoding="utf-8"), "preservar")
            self.assertEqual(json.loads((segunda / "execucao.json").read_text(encoding="utf-8"))["tipo"], "verificacao_funcional")

    def test_avaliador_recuperacao_calcula_por_id_sem_mapeamento_de_nomes(self):
        pergunta = {"id": "Q1", "pergunta": "Pergunta de ensaio", "doc_id": "D02", "pagina_pdf": "6",
                    "expected_action": "responder", "split": "validacao", "tipo": "sintomas", "estado_validacao": "ensaio"}
        passagens = [{"doc_id": doc, "nome_documento": "nome_sem_id.pdf", "pagina": pagina, "score": score}
                     for doc, pagina, score in (("D01", 6, 0.9), ("D02", 5, 0.8), ("D02", 6, 0.7))]
        with tempfile.TemporaryDirectory() as tmp:
            saida = Path(tmp) / "recuperacao.csv"
            with patch("sys.argv", ["recuperacao", "--reranker", "--saida", str(saida)]), \
                    patch.object(avaliar_recuperacao, "resolver_caminho", return_value=Path("ensaio.csv")), \
                    patch.object(avaliar_recuperacao, "carregar_perguntas", return_value=[pergunta]), \
                    patch.object(avaliar_recuperacao, "obter_cliente", return_value=object()), \
                    patch.object(avaliar_recuperacao, "descrever_colecao", return_value={"documentos": ["D01", "D02"]}), \
                    patch.object(avaliar_recuperacao, "pesquisar_query_qdrant", return_value=passagens):
                avaliar_recuperacao.main()
            linha = metricas.csv_ler(saida)[0]
            self.assertEqual(linha["rank_documento"], "2")
            self.assertEqual(linha["rank_pág ±0"], "3")
            self.assertEqual(linha["rank_pág ±1"], "2")

    def test_comparacao_visual_conserva_as_quatro_classes_de_folhas(self):
        manifesto = apoio.ler_json(apoio.FICHEIRO_SPLITS)
        validar_manifesto(manifesto, [1])
        # Colocar Outro primeiro torna visível uma eventual inclusão indevida na comparação.
        manifesto["imagens"].sort(key=lambda i: i["classe"] != "outro")
        folha = next(i for i in manifesto["imagens"]
                     if i["classe"] != "outro" and i["splits"]["1"] == "validacao")
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            foto = raiz / folha["id"]
            foto.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "green").save(foto)
            folha["sha256"] = sha256(foto)
            apoio.gravar_json(raiz / "splits.json", manifesto)
            args = argparse.Namespace(splits=raiz / "splits.json", seeds=[1], split="validacao",
                                      limite=1, modelos=["vlm"], imagens=raiz, execucao="folhas")
            visual = Visual(motor="VLM", estado="ok", classe=folha["classe"], confianca=0.8)
            with patch.object(avaliar, "RUNS", raiz), patch.object(avaliar, "digest_modelo", return_value="ensaio"), \
                    patch.object(avaliar, "analisar_vlm", return_value=visual) as classificar:
                avaliar.avaliar_imagem(args)
            classificar.assert_called_once()
            item_guardado = apoio.ler_json(next((raiz / "folhas/itens").glob("*.json")))
            self.assertEqual(item_guardado["imagem"], folha["id"])

    def test_cli_textual_inclui_tres_condicoes(self):
        with patch("sys.argv", ["avaliar", "texto", "--execucao", "ensaio"]), \
                patch.object(avaliar, "avaliar_texto") as executar:
            avaliar.main()
        self.assertEqual(executar.call_args.args[0].condicoes, list(avaliar.CONDICOES_TEXTO))

    def test_cnn_nao_recebe_imagem_nem_sinais_inventados_no_prompt(self):
        visual = Visual(motor="CNN", estado="ok", classe="esca", confianca=0.8)
        mensagens = rag.mensagens_resposta(carregar_config("llm_cnn"), "Que sinais vês?", [],
                                          visual, Image.new("RGB", (8, 8)), {})
        self.assertIn("Não recebes a fotografia", mensagens[0].content)
        self.assertIn('"sinais_observados": null', mensagens[-1].content)
        self.assertIsInstance(mensagens[-1].content, str)

    def test_regra_documental_fica_identificada_sem_se_fazer_passar_por_pesquisa(self):
        ambito = rag.Ambito(tema="viticultura", seguimento=False, usa_imagem=False, consulta="Edital 2/2025")
        with patch.object(rag, "decidir_ambito", return_value=ambito), patch.object(rag, "recuperar") as pesquisar:
            r = rag.responder("Segundo o Edital 2/2025, que medidas se aplicam à vinha?", "llm_cnn_rag", registar=True)
        self.assertEqual(r.motivo, "documento_ausente")
        self.assertEqual(r.chamadas[-1]["origem"], "metadados_corpus")
        self.assertTrue(r.chamadas[-1]["pesquisa_omitida"])
        pesquisar.assert_not_called()

    def test_quatro_modos_mesma_foto_sem_historico_e_folhas_de_anotacao(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            foto, checkpoint = raiz / "folha.png", raiz / "cnn.keras"
            Image.new("RGB", (8, 8), "green").save(foto)
            checkpoint.write_bytes(b"checkpoint de ensaio, sem pesos")
            cfg_cnn = avaliar.config_cnn().model_copy(update={"checkpoint": str(checkpoint)})
            modos = ["llm_cnn", "llm_cnn_rag", "vlm", "vlm_rag"]
            args = argparse.Namespace(foto=foto, pergunta="Que sinais devo confirmar?", execucao="foto", condicoes=modos)
            resposta = rag.Resposta(estado="resposta", texto="É uma hipótese.", partes=[
                {"texto": "É uma hipótese.", "fontes": [], "trechos": []},
            ])
            with patch.object(avaliar, "RUNS", raiz), patch.object(avaliar, "config_cnn", return_value=cfg_cnn), \
                    patch.object(avaliar, "digest_modelo", return_value="digest de ensaio"), \
                    patch.object(avaliar, "descrever_colecao", return_value={"pontos": 1}), \
                    patch.object(avaliar, "responder", return_value=resposta) as responder:
                avaliar.avaliar_respostas_imagem(args)
            chamadas = responder.call_args_list
            self.assertEqual([c.args[1] for c in chamadas], modos)
            for c in chamadas:
                self.assertEqual(c.args[0], args.pergunta)
                self.assertEqual(set(c.kwargs), {"historico", "imagem", "config", "registar"})
                self.assertEqual(c.kwargs["historico"], ())
                self.assertIs(c.kwargs["imagem"], chamadas[0].kwargs["imagem"])
            ficheiros = list((raiz / "foto/itens").glob("*.json"))
            self.assertEqual(len(ficheiros), 4)
            itens = [json.loads(f.read_text()) for f in ficheiros]
            self.assertEqual({i["imagem_sha256"] for i in itens}, {sha256(foto)})
            self.assertEqual({i["modo"] for i in itens if i["imagem_na_geracao_final"]}, {"vlm", "vlm_rag"})
            manifesto = json.loads((raiz / "foto/execucao.json").read_text())
            with patch.object(metricas, "carregar", return_value=(manifesto, itens)):
                metricas.preparar_anotacao("foto", raiz / "anotacoes")
            linhas = metricas.csv_ler(raiz / "anotacoes/foto/anotador_1.csv")
            self.assertEqual(len(linhas), 4)
            self.assertEqual({r["imagem_sha256"] for r in linhas}, {sha256(foto)})
            tabela = metricas.tabela_respostas_imagem(manifesto, itens, raiz / "anotacoes")
            self.assertEqual({r["condicao"] for r in tabela}, set(modos))
            self.assertNotIn("acuracia", {r["metrica"] for r in tabela})
            self.assertNotIn("fidelidade", {r["metrica"] for r in tabela})
            # Uma anotação real desbloqueia a métrica; uma folha vazia nunca produz zero fictício.
            linhas[0]["suportada"] = "N"
            linhas[0]["resposta_correta"] = "S"
            metricas.csv_gravar(raiz / "anotacoes/foto/consenso.csv", metricas.COLUNAS_ANOTACAO, linhas)
            tabela = metricas.tabela_respostas_imagem(manifesto, itens, raiz / "anotacoes")
            self.assertEqual([r["valor"] for r in tabela if r["metrica"] == "fidelidade"], [0])
            self.assertEqual([r["valor"] for r in tabela if r["metrica"] == "correcao_humana"], [1])

    def test_vlm_sem_rag_nao_exige_checkpoint_ou_qdrant(self):
        with tempfile.TemporaryDirectory() as tmp:
            foto = Path(tmp) / "folha.png"
            Image.new("RGB", (8, 8)).save(foto)
            args = argparse.Namespace(foto=foto, pergunta="Folha?", execucao="vlm", condicoes=["vlm"])
            with patch.object(avaliar, "RUNS", Path(tmp)), \
                    patch.object(avaliar, "config_cnn", side_effect=AssertionError("CNN chamada")), \
                    patch.object(avaliar, "descrever_colecao", side_effect=AssertionError("Qdrant chamado")), \
                    patch.object(avaliar, "digest_modelo", return_value="ensaio"), \
                    patch.object(avaliar, "responder", return_value=rag.Resposta(estado="recusa", texto="Ensaio")):
                avaliar.avaliar_respostas_imagem(args)


if __name__ == "__main__":
    unittest.main()
