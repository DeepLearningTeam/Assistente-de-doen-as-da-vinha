"""Persistência e investigação de conversas, sem chamadas ao Ollama."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import httpx
from fastapi import FastAPI
from gradio.state_holder import SessionState

from app.app import chat_gradio, criar_app, criar_tema, html_modelos
from src import chats
from src.apoio import ErroProjeto
from src.config import carregar_config
from src.rag import Resposta


def conversa_exemplo():
    fonte = {
        "doc": "D02", "pagina": 6, "data": "2007", "titulo": "Doenças do lenho",
        "entidade": "ADVID", "url": "https://example.org/lenho.pdf", "contexto": "Portugal",
        "excerto": "O fungo entra pelas feridas.",
    }
    resposta = Resposta(
        estado="resposta", texto="Pode entrar pelas feridas de poda.",
        partes=[{"texto": "Pode entrar pelas feridas de poda.", "fontes": ["D02:6"]}], fontes=[fonte],
    ).model_dump()
    return {
        **chats.novo_chat(),
        "turnos": [{
            "pergunta": "Como entra a eutipiose?", "modo": "llm_cnn_rag", "com_foto": False,
            "foto": None, "modelos": {"resposta": "modelo-original:1b"}, "resposta": resposta,
        }],
    }


class PastaConversasTest(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.TemporaryDirectory()
        self.addCleanup(self.pasta.cleanup)
        self.patch_pasta = patch.object(chats, "PASTA_CHATS", Path(self.pasta.name))
        self.patch_pasta.start()
        self.addCleanup(self.patch_pasta.stop)


class ConversasTest(PastaConversasTest):
    def test_nome_persiste_ao_renomear_e_continuar(self):
        original = conversa_exemplo()
        original["titulo"] = "  Ensaio   eutipiose  "
        guardado = chats.guardar(original)
        self.assertEqual(chats.abrir(original["id"])["titulo"], "Ensaio eutipiose")
        guardado["titulo"] = "Investigação de novembro"
        guardado["turnos"].append(copy.deepcopy(guardado["turnos"][0]))
        chats.guardar(guardado)
        reaberto = chats.abrir(original["id"])
        self.assertEqual(reaberto["titulo"], "Investigação de novembro")
        self.assertEqual(len(reaberto["turnos"]), 2)
        self.assertEqual(len(list(Path(self.pasta.name).glob("*.json"))), 1)

    def test_nome_automatico_e_conversa_antiga(self):
        chat = conversa_exemplo()
        del chat["titulo"]
        del chat["turnos"][0]["modelos"]
        chat = chats.guardar(chat)
        self.assertEqual(chat["titulo"], chat["turnos"][0]["pergunta"])
        self.assertIn("Não registados nesta conversa antiga", chats.exportar_texto(chats.abrir(chat["id"])))
        self.assertEqual(chats.registos()[0]["modelos"], "Não registado")

    def test_pesquisa_texto_modo_modelo_e_ficheiro_invalido(self):
        chat = chats.guardar(conversa_exemplo())
        (Path(self.pasta.name) / "invalido.json").write_text("{", encoding="utf-8")
        for termo in ("EUTIPIOSE", "feridas de poda", "LLM + CNN + RAG", "modelo-original:1b"):
            with self.subTest(termo=termo):
                self.assertEqual([r["id"] for r in chats.registos(termo)], [chat["id"]])
        self.assertEqual(chats.registos("inexistente"), [])

    def test_exportacao_integral_preserva_fontes_e_exclui_imagens(self):
        chat = conversa_exemplo()
        segundo = copy.deepcopy(chat["turnos"][0])
        segundo.update(modo="vlm", com_foto=True, foto="BYTES_FOTO", modelos={"resposta": "outro-modelo:3b"})
        segundo["resposta"].update(
            estado="recusa", motivo="evidencia_insuficiente",
            visual={"motor": "VLM", "descricao": "Lesões na folha", "mapa_atencao": "BYTES_MAPA"},
        )
        chat["turnos"].append(segundo)
        texto = chats.exportar_texto(chat)
        for trecho in ("Turno 1", "Turno 2", "D02:6", "https://example.org/lenho.pdf",
                       "modelo-original:1b", "outro-modelo:3b", "evidencia_insuficiente", "Lesões na folha"):
            self.assertIn(trecho, texto)
        self.assertNotIn("BYTES_FOTO", texto)
        self.assertNotIn("BYTES_MAPA", texto)
        self.assertIn("mapa_atencao", segundo["resposta"]["visual"])

    def test_conversa_vazia_e_identificador_invalido(self):
        self.assertEqual(chats.exportar_texto(chats.novo_chat()), "")
        with self.assertRaises(ErroProjeto):
            chats.guardar(chats.novo_chat())
        with self.assertRaises(ErroProjeto):
            chats.abrir("../fora")


class InterfaceTest(PastaConversasTest):
    @classmethod
    def setUpClass(cls):
        cls.app = criar_app()
        cls.eventos = {f.fn.__name__: f for f in cls.app.fns.values() if f.fn}

    def test_modelo_corresponde_a_cada_modo(self):
        for modo in ("llm_cnn_rag", "llm_cnn", "vlm", "vlm_rag"):
            with self.subTest(modo=modo):
                config = carregar_config(modo)
                texto = html_modelos(modo)
                self.assertIn(config.modelo, texto)
                self.assertEqual("Embeddings RAG" in texto, config.rag)
                self.assertEqual("CNN:" in texto, config.gerador == "llm")

    def test_modo_e_modelo_de_cada_resposta_persistem_ao_alternar(self):
        chat = conversa_exemplo()
        segundo = copy.deepcopy(chat["turnos"][0])
        segundo.update(modo="vlm", modelos={"resposta": "modelo-visual:3b"})
        segundo["resposta"].update(estado="recusa", texto="Sem suporte documental.")
        chat["turnos"].append(segundo)
        chat["modo"] = "llm_cnn"
        mensagens = chat_gradio(chats.abrir(chats.guardar(chat)["id"]))
        self.assertIn("Modo: LLM + CNN + RAG", mensagens[1]["content"])
        self.assertIn("modelo-original:1b", mensagens[1]["content"])
        self.assertIn("Modo: VLM", mensagens[3]["content"])
        self.assertIn("modelo-visual:3b", mensagens[3]["content"])
        self.assertNotIn("modelo-visual:3b", mensagens[1]["content"])
        del chat["turnos"][0]["modelos"]
        self.assertIn("Modelo não registado", chat_gradio(chat)[1]["content"])

    def test_enviar_guarda_modelo_e_nome_renomear_e_reabrir(self):
        evento = self.eventos["enviar_turno"]
        chat = {**chats.novo_chat(), "titulo": "Ensaio"}
        resposta = Resposta(estado="resposta", texto="Olá. Como posso ajudar?")
        with patch("app.app.responder", return_value=resposta) as responder:
            resultados = list(evento.fn(chat, "Olá", None, "vlm", False))
        estado = evento.outputs[0]
        guardado = resultados[-1][estado]
        self.assertEqual(guardado["titulo"], "Ensaio")
        self.assertEqual(chats.abrir(guardado["id"]), guardado)
        self.assertEqual(guardado["turnos"][0]["modelos"]["resposta"], carregar_config("vlm").modelo)
        self.assertEqual(responder.call_args.kwargs["config"].modelo, carregar_config("vlm").modelo)
        self.assertEqual(chat["turnos"], [])

        atualizado = self.eventos["aplicar_nome"].fn(guardado, guardado["id"], "Renomeada")
        self.assertEqual(atualizado[estado]["titulo"], "Renomeada")
        self.assertEqual(chats.abrir(guardado["id"])["titulo"], "Renomeada")
        reaberto_lateral = self.eventos["abrir_da_lateral"].fn(guardado["id"])
        self.assertEqual(reaberto_lateral[estado]["titulo"], "Renomeada")
        self.assertEqual(reaberto_lateral[evento.outputs[-1]]["value"], guardado["id"])
        self.assertEqual(reaberto_lateral[self.eventos["abrir_da_lateral"].outputs[-2]], "vlm")

    def test_falha_de_gravacao_preserva_resposta_na_sessao(self):
        evento = self.eventos["enviar_turno"]
        with patch("app.app.responder", return_value=Resposta(estado="social", texto="Olá")), \
                patch.object(chats, "guardar", side_effect=OSError("disco indisponível")):
            resultado = list(evento.fn(chats.novo_chat(), "Olá", None, "vlm", False))[-1]
        self.assertEqual(resultado[evento.outputs[0]]["turnos"][0]["resposta"]["texto"], "Olá")
        self.assertIn("não foi possível guardar", resultado[evento.outputs[9]])

    def test_aplicar_nome_antes_da_primeira_mensagem(self):
        evento = self.eventos["aplicar_nome"]
        resultado = evento.fn(chats.novo_chat(), None, "  Ensaio  inicial  ")
        self.assertEqual(resultado[evento.outputs[0]]["titulo"], "Ensaio inicial")
        self.assertEqual(chats.registos(), [])

    def test_renomear_selecionada_sem_alterar_conversa_aberta(self):
        aberta = chats.guardar({**conversa_exemplo(), "titulo": "Em curso"})
        selecionada = chats.guardar({**conversa_exemplo(), "titulo": "No histórico"})
        escolha = self.eventos["selecionar_nome"]
        self.assertEqual(escolha.fn(selecionada["id"])[escolha.outputs[0]], "No histórico")
        evento = self.eventos["aplicar_nome"]
        resultado = evento.fn(aberta, selecionada["id"], "Novo nome")
        self.assertNotIn(evento.outputs[0], resultado)
        self.assertEqual(chats.abrir(selecionada["id"])["titulo"], "Novo nome")
        self.assertEqual(chats.abrir(selecionada["id"])["turnos"], selecionada["turnos"])
        self.assertEqual(chats.abrir(aberta["id"]), aberta)
        self.assertEqual(resultado[evento.outputs[2]]["value"], selecionada["id"])
        envio = self.eventos["enviar_turno"]
        with patch("app.app.responder", return_value=Resposta(estado="social", texto="Olá")):
            list(envio.fn(aberta, "Olá", None, "vlm", False))
        self.assertEqual(chats.abrir(aberta["id"])["titulo"], "Em curso")
        self.assertEqual(chats.abrir(selecionada["id"])["titulo"], "Novo nome")

    def test_nome_vazio_nao_altera_registo(self):
        chat = chats.guardar(conversa_exemplo())
        evento = self.eventos["aplicar_nome"]
        resultado = evento.fn(chats.novo_chat(), chat["id"], "  ")
        self.assertIn("Escreve o novo nome", resultado[evento.outputs[-1]])
        self.assertEqual(chats.abrir(chat["id"]), chat)


class InterfaceIntegracaoTest(unittest.IsolatedAsyncioTestCase):
    async def test_eventos_gradio_e_paginas_http(self):
        with tempfile.TemporaryDirectory() as pasta, patch.object(chats, "PASTA_CHATS", Path(pasta)):
            app = criar_app().queue(default_concurrency_limit=1)
            servidor = gr.mount_gradio_app(
                FastAPI(), app, path="/", theme=criar_tema(), footer_links=["runs"], run_history=True,
            )
            eventos = {f.fn.__name__: f for f in app.fns.values() if f.fn}
            sessao = SessionState(app)
            enviar = eventos["enviar_turno"]
            sessao[enviar.inputs[0]._id] = {**chats.novo_chat(), "titulo": "Ensaio HTTP"}
            resposta = Resposta(estado="social", texto="Olá. Como posso ajudar?")
            with patch("app.app.responder", return_value=resposta):
                retorno = await app.process_api(
                    enviar, [None, "Olá", None, "vlm", False], state=sessao,
                )
                while retorno["is_generating"]:
                    retorno = await app.process_api(enviar, [], state=sessao, iterator=retorno["iterator"])
            chat = sessao[enviar.inputs[0]._id]
            self.assertEqual(chats.abrir(chat["id"]), chat)
            retorno = await app.process_api(eventos["abrir_da_lateral"], [chat["id"]], state=sessao)
            self.assertEqual(len(retorno["data"][1]), 2)
            self.assertEqual(retorno["data"][-2], "vlm")
            transporte = httpx.ASGITransport(app=servidor)
            async with httpx.AsyncClient(transport=transporte, base_url="http://testserver") as cliente:
                pagina = await cliente.get("/")
                config = await cliente.get("/config")
                runs = await cliente.get("/gradio_api/runs")
            self.assertEqual(pagina.status_code, 200)
            self.assertEqual(config.status_code, 200)
            self.assertEqual(runs.status_code, 200)
            self.assertEqual(config.json()["footer_links"], ["runs"])
            self.assertTrue(config.json()["run_history"])
            envio = next(d for d in config.json()["dependencies"] if d["api_name"] == "enviar_turno")
            self.assertEqual(envio["api_visibility"], "public")


if __name__ == "__main__":
    unittest.main()
