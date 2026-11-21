"""Contrato comum da geração de respostas com e sem fotografia ou documentos."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from src import rag
from src.config import carregar_config
from src.imagem import Visual


CONDICOES = ("llm_cnn", "llm_rag_sem_limiar", "llm_cnn_rag")
MODOS = (*CONDICOES, "vlm", "vlm_rag")
PERGUNTA = "Por onde pode entrar o fungo associado à eutipiose?"
AMBITO = rag.Ambito(tema="viticultura", seguimento=False, usa_imagem=False, consulta=PERGUNTA)
PASSAGEM = {
    "id": "ensaio:6", "doc": "D02", "pagina": 6, "titulo": "Documento de ensaio",
    "data": "2020", "entidade": "Ensaio", "contexto": "Portugal",
    "url": "https://example.org/documento.pdf", "similaridade": 0.9,
    "texto": "O fungo pode entrar por feridas de poda.",
}


class ComparacaoGeracao(unittest.TestCase):
    def test_mesmo_sistema_esquema_e_orcamento_no_cliente_do_modelo(self):
        # Exercita o percurso público e o adaptador Ollama, simulando só o serviço.
        for limite in (128, 637):
            sistemas, esquemas, parametros = [], [], []
            for modo in CONDICOES:
                config = carregar_config(modo).model_copy(update={"max_resposta_tokens": limite})
                with self.subTest(modo=modo, limite=limite), \
                        patch.object(rag, "decidir_ambito", return_value=AMBITO), \
                        patch.object(rag, "recuperar", return_value=[PASSAGEM]) as recuperar, \
                        patch.object(rag, "analisar_cnn") as cnn, \
                        patch.object(rag, "confirmar_suporte", return_value=True), \
                        patch("src.modelos.cliente_chat") as cliente:
                    geracao = {"primeira": 1 if config.rag else 0, "segunda": 0,
                               "resposta": "O fungo pode entrar por feridas de poda."}
                    cliente.return_value.bind.return_value.invoke.return_value = SimpleNamespace(
                        content=json.dumps(geracao),
                    )
                    resposta = rag.responder(PERGUNTA, modo, config=config)
                    self.assertEqual(resposta.estado, "resposta")
                    self.assertEqual(bool(resposta.fontes), config.rag)
                    cnn.assert_not_called()
                    self.assertEqual(recuperar.call_count, int(config.rag))
                    cliente.assert_called_once()
                    self.assertEqual(cliente.call_args.args[-1], limite)
                    parametros.append(cliente.call_args.args)
                    mensagens = cliente.return_value.bind.return_value.invoke.call_args.args[0]
                    sistemas.append(mensagens[0].content)
                    self.assertTrue(mensagens[-1].content.startswith(PERGUNTA))
                    self.assertEqual("Passagens de referência:" in mensagens[-1].content, config.rag)
                    esquemas.append(cliente.return_value.bind.call_args.kwargs["format"])
            self.assertTrue(all(s == sistemas[0] for s in sistemas))
            self.assertTrue(all(e == esquemas[0] for e in esquemas))
            self.assertTrue(all(p == parametros[0] for p in parametros))
            self.assertEqual(set(esquemas[0]["properties"]), {"primeira", "segunda", "resposta"})

    def test_sistema_comum_preservado_na_reparacao_e_revisao(self):
        sistemas = []
        for modo in MODOS:
            config = carregar_config(modo)
            fontes = [1] if config.rag else []
            completa = ([rag.Parte("O fungo pode entrar por feridas de poda.", fontes)], "")
            tentativas = [([], "incompleta"), completa]
            if config.recusa_reforcada:
                tentativas.append(completa)
            with self.subTest(modo=modo), \
                    patch.object(rag, "decidir_ambito", return_value=AMBITO), \
                    patch.object(rag, "recuperar", return_value=[PASSAGEM]), \
                    patch.object(rag, "gerar_partes", side_effect=tentativas) as gerar, \
                    patch.object(rag, "verificar_adequacao", return_value=True), \
                    patch.object(rag, "confirmar_suporte", side_effect=[False, True]):
                resposta = rag.responder(PERGUNTA, modo, config=config)
            self.assertEqual(resposta.estado, "resposta")
            for chamada in gerar.call_args_list:
                mensagens = chamada.args[1]
                self.assertEqual(sum(m.type == "system" for m in mensagens), 1)
                sistemas.append(mensagens[0].content)
            self.assertEqual(gerar.call_count, 3 if config.recusa_reforcada else 2)
        self.assertTrue(all(s == sistemas[0] for s in sistemas))

    def test_geracoes_adicionais_usam_o_mesmo_limite_configurado(self):
        for modo in MODOS:
            config = carregar_config(modo).model_copy(update={"max_resposta_tokens": 731})
            for etapa in ("geracao", "completar_resposta", "revisao"):
                with self.subTest(modo=modo, etapa=etapa), patch.object(rag, "conversar") as modelo:
                    modelo.return_value = rag.GeracaoDocumental(
                        primeira=1 if config.rag else 0, segunda=0, resposta="Uma resposta de ensaio.",
                    )
                    partes, motivo = rag.gerar_partes(config, [], {1: PASSAGEM}, None, etapa=etapa)
                self.assertTrue(partes)
                self.assertEqual(motivo, "")
                self.assertIs(modelo.call_args.args[2], rag.GeracaoDocumental)
                self.assertEqual(modelo.call_args.kwargs["limite"], 731)

    def test_fotografia_partilha_sistema_esquema_e_limite(self):
        imagem = Image.new("RGB", (8, 8), "green")
        sistemas, esquemas, fotografias = [], [], []
        for modo in MODOS:
            config = carregar_config(modo).model_copy(update={"max_resposta_tokens": 637})
            motor = "VLM" if config.gerador == "vlm" else "CNN"
            visual = Visual(motor=motor, estado="ok", classe="esca", confianca=0.8)
            with self.subTest(modo=modo), \
                    patch.object(rag, "decidir_ambito", return_value=AMBITO.model_copy(update={"usa_imagem": True})), \
                    patch.object(rag, "analisar_cnn", return_value=visual), \
                    patch.object(rag, "analisar_vlm", return_value=visual), \
                    patch.object(rag, "recuperar", return_value=[PASSAGEM]), \
                    patch.object(rag, "confirmar_suporte", return_value=True), \
                    patch("src.modelos.cliente_chat") as cliente:
                cliente.return_value.bind.return_value.invoke.return_value = SimpleNamespace(content=json.dumps({
                    "primeira": 1 if config.rag else 0, "segunda": 0, "resposta": "Ensaio.",
                }))
                resposta = rag.responder("Que sinais se observam?", modo, imagem=imagem, config=config)
            self.assertEqual(resposta.estado, "resposta")
            self.assertEqual(cliente.call_args.args[-1], 637)
            mensagens = cliente.return_value.bind.return_value.invoke.call_args.args[0]
            sistemas.append(mensagens[0].content)
            esquemas.append(cliente.return_value.bind.call_args.kwargs["format"])
            if motor == "VLM":
                fotografias.append(mensagens[-1].content[1])
            else:
                self.assertIsInstance(mensagens[-1].content, str)
        self.assertTrue(all(s == sistemas[0] for s in sistemas))
        self.assertTrue(all(e == esquemas[0] for e in esquemas))
        self.assertEqual(len(fotografias), 2)
        self.assertEqual(fotografias[0], fotografias[1])

    def test_contexto_visual_identico_no_par_com_e_sem_rag(self):
        imagem = Image.new("RGB", (8, 8), "green")
        for motor, par in (("CNN", ("llm_cnn", "llm_cnn_rag")), ("VLM", ("vlm", "vlm_rag"))):
            visual = Visual(motor=motor, estado="ok", classe="esca", confianca=0.8,
                            descricao="Folha com manchas.")
            pedidos = []
            for modo in par:
                config = carregar_config(modo)
                mensagens = rag.mensagens_resposta(config, PERGUNTA, [], visual, imagem,
                                                   {1: PASSAGEM} if config.rag else {})
                conteudo = mensagens[-1].content
                texto = conteudo[0]["text"] if isinstance(conteudo, list) else conteudo
                pedidos.append(texto.split("\n\nPassagens de referência:")[0])
            self.assertEqual(pedidos[0], pedidos[1])
            self.assertIn("Folha com manchas.", pedidos[0])

    def test_reparacao_visual_preserva_sistema_e_fotografia(self):
        imagem = Image.new("RGB", (8, 8), "green")
        visual = Visual(motor="VLM", estado="ok", classe="esca", confianca=0.8)
        completa = ([rag.Parte("Uma resposta de ensaio.", [1])], "")
        with patch.object(rag, "decidir_ambito", return_value=AMBITO.model_copy(update={"usa_imagem": True})), \
                patch.object(rag, "analisar_vlm", return_value=visual), \
                patch.object(rag, "recuperar", return_value=[PASSAGEM]), \
                patch.object(rag, "gerar_partes", side_effect=[([], "incompleta"), completa, completa]) as gerar, \
                patch.object(rag, "confirmar_suporte", side_effect=[False, True]):
            resposta = rag.responder("Que sinais se observam?", "vlm_rag", imagem=imagem)
        self.assertEqual(resposta.estado, "resposta")
        self.assertEqual(gerar.call_count, 3)
        sistemas, fotografias = [], []
        for chamada in gerar.call_args_list:
            mensagens = chamada.args[1]
            sistemas.append([m.content for m in mensagens if m.type == "system"])
            imagens = [parte for m in mensagens if isinstance(m.content, list)
                       for parte in m.content if parte["type"] == "image_url"]
            self.assertEqual(len(imagens), 1)
            fotografias.append(imagens[0])
        self.assertEqual(sistemas, [sistemas[0]] * 3)
        self.assertEqual(len(sistemas[0]), 1)
        self.assertEqual(fotografias, [fotografias[0]] * 3)


if __name__ == "__main__":
    unittest.main()
