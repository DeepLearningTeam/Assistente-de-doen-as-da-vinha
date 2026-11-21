"""Respostas completas e reparação limitada sem inventar citações."""
import unittest
from unittest.mock import patch

from src.config import carregar_config
from src.rag import Ambito, GeracaoDocumental, Parte, Resposta, decidir_ambito, gerar_partes, responder


AMBITO = Ambito(tema="viticultura", seguimento=False, usa_imagem=False, consulta="Quais são as doenças da videira?")
PASSAGEM = {
    "id": "D01:1:1", "doc": "D01", "pagina": 1, "data": "2020", "contexto": "Portugal",
    "titulo": "Documento de ensaio", "entidade": "Ensaio", "url": "https://example.org/ensaio.pdf",
    "texto": "Este documento apresenta exemplos de doenças.", "similaridade": 0.8,
}


class RespostasTest(unittest.TestCase):
    def test_introducao_sem_lista_nao_e_resposta_valida(self):
        config = carregar_config("llm_cnn")
        for texto in ("Alguns exemplos incluem:", "Exemplos:\n-", "Exemplos:\n1."):
            geracao = GeracaoDocumental(primeira=0, segunda=0, resposta=texto)
            with self.subTest(texto=texto), patch("src.rag.conversar", return_value=geracao):
                partes, motivo = gerar_partes(config, [], {}, None)
            self.assertEqual(partes, [])
            self.assertEqual(motivo, "incompleta")

    def test_texto_documental_sem_citacao_e_diferente_de_sem_fonte(self):
        config = carregar_config("llm_cnn_rag")
        with patch("src.rag.conversar", return_value=GeracaoDocumental(primeira=0, segunda=0, resposta="Há vários exemplos.")):
            _, motivo = gerar_partes(config, [], {1: PASSAGEM}, None)
        self.assertEqual(motivo, "citacoes_ausentes")
        with patch("src.rag.conversar", return_value=GeracaoDocumental(primeira=0, segunda=0, resposta="SEM_FONTE")):
            _, motivo = gerar_partes(config, [], {1: PASSAGEM}, None)
        self.assertEqual(motivo, "geracao")

    def test_resposta_incompleta_e_reparada_uma_vez(self):
        with patch("src.rag.decidir_ambito", return_value=AMBITO), patch("src.rag.gerar_partes", side_effect=[
            ([], "incompleta"), ([Parte("Exemplos completos de ensaio.", [])], ""),
        ]) as gerar, patch("src.rag.verificar_adequacao", return_value=True):
            resposta = responder("Quais são as doenças?", "llm_cnn")
        self.assertEqual(resposta.estado, "resposta")
        self.assertEqual(resposta.texto, "Exemplos completos de ensaio.")
        self.assertEqual(gerar.call_count, 2)
        self.assertEqual(gerar.call_args.kwargs["etapa"], "completar_resposta")

    def test_reparacao_nao_repete_indefinidamente_nem_publica_fragmento(self):
        with patch("src.rag.decidir_ambito", return_value=AMBITO), \
                patch("src.rag.gerar_partes", return_value=([], "incompleta")) as gerar:
            resposta = responder("Quais são as doenças?", "llm_cnn")
        self.assertEqual(resposta.estado, "recusa")
        self.assertEqual(resposta.motivo, "incompleta")
        self.assertEqual(gerar.call_count, 2)
        self.assertNotIn("fontes consultadas", resposta.texto)

    def test_citacoes_em_falta_sao_reparadas_e_verificadas(self):
        with patch("src.rag.decidir_ambito", return_value=AMBITO), \
                patch("src.rag.recuperar", return_value=[PASSAGEM]), \
                patch("src.rag.gerar_partes", side_effect=[
                    ([], "citacoes_ausentes"), ([Parte("Este documento apresenta exemplos de doenças.", [1])], ""),
                ]), patch("src.rag.confirmar_suporte", return_value=True) as verificar:
            resposta = responder("Quais são as doenças?", "llm_cnn_rag")
        self.assertEqual(resposta.estado, "resposta")
        self.assertEqual(resposta.fontes[0]["doc"], "D01")
        verificar.assert_called_once()

    def test_reparacao_inadequada_nao_e_publicada(self):
        with patch("src.rag.decidir_ambito", return_value=AMBITO), patch("src.rag.gerar_partes", side_effect=[
            ([], "incompleta"), ([Parte("Lista incoerente de ensaio.", [])], ""),
        ]), patch("src.rag.verificar_adequacao", return_value=False):
            resposta = responder("Quais são as doenças?", "llm_cnn")
        self.assertEqual(resposta.estado, "recusa")
        self.assertEqual(resposta.motivo, "resposta_inadequada")
        self.assertNotIn("Lista incoerente", resposta.texto)

    def test_pergunta_autonoma_nao_reutiliza_resposta_visual_anterior(self):
        anterior = {"pergunta": "Que doença tem esta folha?", "modo": "vlm", "resposta": Resposta(
            estado="resposta", texto="Hipótese anterior que não pertence à pergunta geral.",
        ).model_dump()}
        with patch("src.rag.decidir_ambito", return_value=AMBITO), patch("src.rag.gerar_partes", return_value=(
            [Parte("Exemplos de ensaio.", [])], "",
        )) as gerar:
            responder("Quais são as diferentes doenças na viticultura?", "llm_cnn", [anterior])
        self.assertEqual(len(gerar.call_args.args[1]), 2)

    def test_pergunta_geral_nao_passa_a_seguimento_por_existir_historico(self):
        historico = [{"pergunta": "Que doença tem esta folha?", "resposta": Resposta(
            estado="resposta", texto="Uma hipótese anterior.",
        ).model_dump()}]
        with patch("src.rag.conversar", return_value=AMBITO) as modelo:
            resultado = decidir_ambito(carregar_config("llm_cnn"), "Quais são as diferentes doenças na viticultura?", historico, False, None)
        self.assertFalse(resultado.seguimento)
        modelo.assert_called_once()


if __name__ == "__main__":
    unittest.main()
