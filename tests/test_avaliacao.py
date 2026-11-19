"""Regressões da avaliação; sem pesos nem serviços remotos."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src import analise, avaliar, metricas, rag
from src.apoio import ErroProjeto, sha256
from src.config import carregar_config
from src.imagem import Visual


def item(chave="r1", **alteracoes):
    return {
        "chave": chave, "estado": "resposta", "fontes": [], "rag": False,
        "referencia": {"expected_action": "responder", "doc_id": "D02", "pagina_pdf": "6"},
        "latencia_s": 1, "top_k": 5, **alteracoes,
    }


class MetricasAvaliacao(unittest.TestCase):
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


class PercursosAvaliacao(unittest.TestCase):
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
                self.assertEqual(c.kwargs["historico"], ())
                self.assertIs(c.kwargs["imagem"], chamadas[0].kwargs["imagem"])
            ficheiros = list((raiz / "foto/itens").glob("*.json"))
            self.assertEqual(len(ficheiros), 4)
            itens = [json.loads(f.read_text()) for f in ficheiros]
            self.assertEqual({i["imagem_sha256"] for i in itens}, {sha256(foto)})
            self.assertEqual([i["modo"] for i in itens if i["imagem_na_geracao_final"]], ["vlm"])
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
