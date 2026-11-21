"""Testes locais: Qdrant real em memória; modelos substituídos por dados de ensaio."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, PointStruct, SparseVector, SparseVectorParams, VectorParams

from src import avaliar, chats, indice, ingestao, rag
from src.apoio import ErroProjeto
from src.config import MODOS, carregar_config
from src.corpus import ler_corpus
from src.imagem import Visual


def passagem(doc="D02", pagina=6, texto=None):
    fonte = ler_corpus()[doc]
    return {**{k: fonte[k] for k in ("titulo", "entidade", "data", "url")},
            "doc_id": doc, "nome_documento": f"{doc}.pdf", "pagina": pagina,
            "estado_fonte": "sim", "cabecalhos": "Eutipiose",
            "texto": texto or "O fungo Eutypa lata entra através de feridas recentes, incluindo feridas de poda."}


class BGEEnsaio:
    def encode(self, textos, **kwargs):
        return np.array([[1.0] + [0.0] * 1023 for _ in textos])


class BM25Ensaio:
    def query_embed(self, texto):
        yield SimpleNamespace(indices=np.array([1]), values=np.array([1.0]))

    def passage_embed(self, textos):
        for _ in textos:
            yield SimpleNamespace(indices=np.array([1]), values=np.array([1.0]))


class RerankerEnsaio:
    def predict(self, pares, **kwargs):
        return np.array([0.95 if "feridas" in pergunta else 0.10 for pergunta, _ in pares])


def modelo_ensaio(config, mensagens, esquema, **kwargs):
    if esquema is rag.Ambito:
        pedido = json.loads(mensagens[-1].content)
        return rag.Ambito(tema="viticultura", seguimento=False, usa_imagem=False,
                          consulta=pedido["mensagem"])
    if esquema is rag.GeracaoDocumental:
        return rag.GeracaoDocumental(primeira=1 if config.rag else 0, segunda=0,
                                     resposta="O fungo pode entrar por feridas de poda.")
    if esquema is rag.Verificacao:
        return rag.Verificacao(suportada=True)
    return rag.GeracaoDocumental(primeira=0, segunda=0, resposta="O fungo pode entrar por feridas de poda.")


class Integracao(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.cfg = carregar_config().model_copy(update={"qdrant_colecao": "ensaio"})
        self.client.create_collection(
            "ensaio", vectors_config={"vetor_denso": VectorParams(size=1024, distance=Distance.COSINE)},
            sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
        )
        pontos = []
        for numero, doc in enumerate(("D02", "D10"), 1):
            pontos.append(PointStruct(id=numero, payload=passagem(doc), vector={
                "vetor_denso": [1.0] + [0.0] * 1023,
                "bm25": SparseVector(indices=[1], values=[1.0]),
            }))
        self.client.upsert("ensaio", pontos)
        patches = (
            patch.object(indice, "obter_cliente", return_value=self.client),
            patch.object(indice, "obter_modelo_bge", return_value=BGEEnsaio()),
            patch.object(indice, "obter_bm25", return_value=BM25Ensaio()),
            patch.object(indice, "obter_reranker", return_value=RerankerEnsaio()),
            patch.object(ingestao, "obter_modelo_bge", return_value=BGEEnsaio()),
            patch.object(ingestao, "obter_bm25", return_value=BM25Ensaio()),
            patch.object(rag, "conversar", side_effect=modelo_ensaio),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.client.close)

    def test_hibrida_recupera_passagem_e_exclui_holdout(self):
        resultados = indice.recuperar(self.cfg, "Por que feridas entra o fungo?")
        self.assertEqual([(r["doc"], r["pagina"]) for r in resultados], [("D02", 6)])
        self.assertIn("feridas de poda", resultados[0]["texto"])
        self.assertEqual(resultados[0]["similaridade"], 0.95)

    def test_resposta_cita_documento_pagina_e_link(self):
        from app.app import chat_gradio
        resposta = rag.responder("Por que feridas entra o fungo?", "llm_cnn_rag", config=self.cfg, registar=True)
        self.assertEqual(resposta.estado, "resposta")
        self.assertEqual(resposta.partes[0]["fontes"], ["D02:6"])
        self.assertEqual(resposta.fontes[0]["pagina"], 6)
        chat = chats.novo_chat()
        chat["turnos"] = [{"pergunta": "Eutipiose?", "modo": "llm_cnn_rag", "resposta": resposta.model_dump()}]
        conteudo = chat_gradio(chat)[-1]["content"]
        self.assertIn("D02, p. 6", conteudo)
        self.assertIn("#page=6", conteudo)

    def test_sem_suporte_recusa_e_conserva_topk_antes_do_corte(self):
        resposta = rag.responder("Qual é o detalhe ausente?", "llm_cnn_rag", config=self.cfg, registar=True)
        self.assertEqual(resposta.estado, "recusa")
        self.assertEqual(resposta.motivo, "similaridade")
        self.assertEqual(resposta.recuperadas[0]["similaridade"], 0.10)

    def test_ablacao_sem_limiar_recebe_score_baixo(self):
        cfg = self.cfg.model_copy(update={"recusa_reforcada": False})
        resposta = rag.responder("Qual é o detalhe ausente?", "llm_rag_sem_limiar", config=cfg)
        self.assertEqual(resposta.estado, "resposta")

    def test_baseline_sem_qdrant_ou_cnn(self):
        cfg = carregar_config("llm_cnn")
        with patch.object(indice, "obter_cliente", side_effect=AssertionError("Qdrant chamado")), \
             patch.object(rag, "analisar_cnn", side_effect=AssertionError("CNN chamada")):
            resposta = rag.responder("Por que feridas entra o fungo?", "llm_cnn", config=cfg)
        self.assertEqual(resposta.estado, "resposta")
        self.assertFalse(resposta.fontes)

    def test_vlm_sem_rag_nao_chama_qdrant(self):
        with patch.object(indice, "obter_cliente", side_effect=AssertionError("Qdrant chamado")):
            resposta = rag.responder("O que é eutipiose?", "vlm", config=carregar_config("vlm"))
        self.assertEqual(resposta.estado, "resposta")

    def test_verificacao_rejeita_afirmacao_sem_suporte(self):
        def rejeitar(config, mensagens, esquema, **kwargs):
            if esquema is rag.Verificacao:
                return rag.Verificacao(suportada=False)
            return modelo_ensaio(config, mensagens, esquema, **kwargs)
        with patch.object(rag, "conversar", side_effect=rejeitar):
            resposta = rag.responder("Por que feridas entra o fungo?", "llm_cnn_rag", config=self.cfg)
        self.assertEqual(resposta.estado, "recusa")
        self.assertEqual(resposta.motivo, "evidencia_insuficiente")

    def test_diagnostico_deteta_colecao_contaminada(self):
        with self.assertRaisesRegex(ErroProjeto, "reservados"):
            indice.descrever_colecao(self.cfg)

    def test_esquema_incompativel(self):
        self.client.create_collection("errada", vectors_config=VectorParams(size=4, distance=Distance.COSINE))
        with self.assertRaisesRegex(ErroProjeto, "vetor_denso"):
            indice.validar_colecao(self.client, self.cfg.model_copy(update={"qdrant_colecao": "errada"}))

    def test_construcao_impede_cliente_remoto(self):
        with self.assertRaisesRegex(ErroProjeto, "limitada"):
            ingestao.construir_base_qdrant([passagem()], SimpleNamespace(_client=object()), self.cfg)

    def test_extracao_real_preserva_paginas(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "ensaio.pdf"
            with pymupdf.open() as pdf:
                for palavra in ("Eutipiose", "Esca"):
                    pagina = pdf.new_page()
                    pagina.insert_textbox((50, 50, 500, 750),
                                          f"{palavra}: Este documento de ensaio contém informação sobre feridas de poda e sintomas da videira.")
                pdf.save(caminho)
            entrada = {**ler_corpus()["D02"], "paginas": "2"}
            with patch.object(ingestao, "obter_pdf", return_value=caminho):
                chunks = ingestao.processar_corpus([entrada], config=self.cfg, ocr="off")
        self.assertEqual({c["pagina"] for c in chunks}, {1, 2})
        self.assertTrue(all(c["doc_id"] == "D02" for c in chunks))

    def test_indexacao_local_e_pesquisa_com_mesmos_campos(self):
        cfg = self.cfg.model_copy(update={"qdrant_colecao": "nova"})
        ingestao.construir_base_qdrant([passagem()], self.client, cfg)
        resultados = indice.recuperar(cfg, "Por que feridas entra o fungo?")
        self.assertEqual(resultados[0]["doc"], "D02")

    def test_cli_ingestao_cria_fecha_e_preserva_colecao_existente(self):
        with tempfile.TemporaryDirectory() as tmp:
            pasta = Path(tmp) / "qdrant"
            with patch("sys.argv", ["ingestao", "--local", str(pasta), "--colecao", "nova", "--ocr", "off"]), \
                 patch.object(ingestao, "ler_corpus", return_value=[ler_corpus()["D02"]]), \
                 patch.object(ingestao, "verificar"), \
                 patch.object(ingestao, "processar_corpus", return_value=[passagem()]):
                ingestao.main()
                with self.assertRaisesRegex(SystemExit, "já existe"):
                    ingestao.main()
            client = QdrantClient(path=str(pasta))
            try:
                self.assertEqual(client.get_collection("nova").points_count, 1)
            finally:
                client.close()

    def test_imagem_na_geracao_final_nos_dois_modos_vlm(self):
        imagem = Image.new("RGB", (8, 8), "green")
        visual = Visual(motor="VLM", estado="ok", classe="esca", confianca=0.8, descricao="Manchas nas folhas.")
        p = indice.normalizar_passagem({**passagem(), "score": 0.95, "id": "1"})
        for modo, tem_foto in (("vlm", True), ("vlm_rag", True)):
            mensagens = rag.mensagens_resposta(carregar_config(modo), "Que sinais vês?", [], visual, imagem, {1: p})
            self.assertEqual(isinstance(mensagens[-1].content, list), tem_foto)

    def test_avaliacao_guarda_referencia_sem_a_enviar(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(avaliar, "RUNS", Path(tmp)), \
             patch.object(avaliar, "digest_modelo", return_value="digest_ensaio"), \
             patch.object(avaliar, "descrever_colecao", return_value={"pontos": 1}), \
             patch.object(avaliar, "responder", return_value=rag.Resposta(estado="resposta", texto="Ensaio")) as responder:
            args = argparse.Namespace(split="validacao", limite=1, condicoes=["llm_cnn", "llm_cnn_rag"],
                                      execucao="ensaio", repeticoes=1, temperaturas=[0])
            avaliar.avaliar_texto(args)
            chamadas = responder.call_args_list
            self.assertEqual(len(chamadas), 2)
            ref = avaliar.ler_perguntas()[0]
            for chamada in chamadas:
                self.assertEqual(chamada.args[0], ref["pergunta"])
                self.assertNotIn("historico", chamada.kwargs)  # valor predefinido: ()
                self.assertNotIn("imagem", chamada.kwargs)
                self.assertNotIn(ref["resposta_ref"], chamada.args[0])
            itens = list((Path(tmp) / "ensaio/itens").glob("*.json"))
            self.assertEqual(len(itens), 2)
            for ficheiro in itens:
                self.assertEqual(json.loads(ficheiro.read_text(encoding="utf-8"))["referencia"], ref)

    def test_historico_entre_modos_e_persistencia(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(chats, "PASTA_CHATS", Path(tmp)):
            chat = chats.novo_chat()
            chat["turnos"] = [{"pergunta": "Eutipiose?", "modo": "llm_cnn_rag",
                               "resposta": rag.Resposta(estado="resposta", texto="Informação RAG.").model_dump()}]
            chat["modo"] = "llm_cnn"
            guardado = chats.guardar(chat)
            aberto = chats.abrir(guardado["id"])
            self.assertEqual(aberto["modo"], "llm_cnn")
            self.assertEqual(rag.historico_util(aberto["turnos"], 6)[0]["resposta"]["texto"], "Informação RAG.")

    def test_gradio_http_e_quatro_modos(self):
        from app.app import criar_app, criar_tema
        import gradio as gr
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        demo = criar_app()
        servidor = gr.mount_gradio_app(FastAPI(), demo, path="/", theme=criar_tema())
        with TestClient(servidor) as client:
            self.assertEqual(client.get("/").status_code, 200)
            resposta = client.get("/config")
            self.assertEqual(resposta.status_code, 200)
            for rotulo in MODOS.values():
                self.assertIn(rotulo, resposta.text)


if __name__ == "__main__":
    unittest.main()
