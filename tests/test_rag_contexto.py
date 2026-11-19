"""Regressões do contexto de fotografia, correção e pergunta de seguimento."""
import json
import unittest
from unittest.mock import patch

from PIL import Image

from src.config import carregar_config
from src.imagem import Visual
from src.rag import Ambito, Parte, Resposta, decidir_ambito, historico_util, responder


def historico_com_correcao():
    visual = Visual(motor="VLM", estado="ok", classe="esca", confianca=0.85, descricao="Manchas na folha.")
    return [
        {"pergunta": "Que doença tem esta folha?", "modo": "vlm_rag", "resposta": Resposta(
            estado="resposta", texto="Hipótese de esca.", visual=visual,
        ).model_dump()},
        {"pergunta": "Fui confirmar e estás errado. É black rot.", "modo": "vlm_rag", "resposta": Resposta(
            estado="correcao", texto="Vou usar a identificação indicada: black rot.", correcao="black rot",
        ).model_dump()},
    ]


PASSAGEM = {
    "id": "D06:2:1", "doc": "D06", "pagina": 2, "data": "2014", "contexto": "Portugal",
    "titulo": "Documento de ensaio sobre black rot", "entidade": "Ensaio",
    "url": "https://example.org/ensaio.pdf", "texto": "O documento descreve a prevenção de black rot.",
    "similaridade": 0.8,
}


class ContextoTest(unittest.TestCase):
    def setUp(self):
        self.config = carregar_config("vlm_rag")

    def test_pedido_viticultura_e_reavaliado_com_historico_mesmo_sem_flag(self):
        primeira = Ambito(tema="viticultura", seguimento=False, usa_imagem=False, consulta="Como tratar a doença?")
        segunda = Ambito(tema="viticultura", seguimento=True, usa_imagem=False, consulta="Como tratar black rot?")
        with patch("src.rag.conversar", side_effect=[primeira, segunda]) as modelo:
            resultado = decidir_ambito(self.config, "Como tratar a doença?", historico_com_correcao(), False, None)
        self.assertEqual(resultado.consulta, "Como tratar black rot?")
        pedido = json.loads(modelo.call_args.args[1][-1].content)
        self.assertEqual(len(pedido["turnos_anteriores"]), 2)

    def test_correcao_e_reconhecida_sem_pesquisa_ou_confirmacao_visual(self):
        ambito = Ambito(tema="viticultura", seguimento=True, usa_imagem=False, consulta="black rot", correcao="black rot")
        with patch("src.rag.decidir_ambito", return_value=ambito), patch("src.rag.recuperar") as pesquisa:
            resposta = responder("Fui confirmar. É black rot.", "vlm_rag", historico_com_correcao()[:1])
        self.assertEqual(resposta.estado, "correcao")
        self.assertEqual(resposta.correcao, "black rot")
        self.assertIn("identificação indicada", resposta.texto)
        self.assertEqual(resposta.fontes, [])
        pesquisa.assert_not_called()
        turno = {"pergunta": "É black rot.", "modo": "vlm_rag", "resposta": resposta.model_dump()}
        self.assertEqual(historico_util([turno], 6), [turno])

    def test_reinterpretacao_nao_apaga_correcao_explicita(self):
        correcao = Ambito(tema="viticultura", seguimento=False, usa_imagem=False, consulta="black rot", correcao="black rot")
        with patch("src.rag.conversar", return_value=correcao) as modelo:
            resultado = decidir_ambito(self.config, "É black rot.", historico_com_correcao()[:1], False, None)
        self.assertEqual(resultado.correcao, "black rot")
        modelo.assert_called_once()

    def test_seguimento_pesquisa_assunto_corrigido_sem_reusar_visual_antigo(self):
        consulta = "Como tratar a podridão-negra (black rot) na videira?"
        ambito = Ambito(tema="viticultura", seguimento=True, usa_imagem=True, consulta=consulta)
        with patch("src.rag.decidir_ambito", return_value=ambito), \
                patch("src.rag.recuperar", return_value=[PASSAGEM]) as pesquisa, \
                patch("src.rag.gerar_partes", return_value=([Parte("O documento descreve a prevenção de black rot.", [1])], "")) as gerar, \
                patch("src.rag.confirmar_suporte", return_value=True):
            resposta = responder("Como tratar a doença que mencionas?", "vlm_rag", historico_com_correcao())
        self.assertEqual(resposta.estado, "resposta")
        self.assertIn("Como tratar a doença que mencionas?", pesquisa.call_args.args[1])
        self.assertIn("black rot", pesquisa.call_args.args[1])
        self.assertNotIn("Esca", resposta.consulta)
        self.assertIn(resposta.consulta, gerar.call_args.args[1][-1].content)
        self.assertNotIn("Hipótese de esca.", "\n".join(m.content for m in gerar.call_args.args[1]))
        self.assertIsNone(resposta.visual)
        self.assertEqual(resposta.fontes[0]["doc"], "D06")

    def test_referencia_ambigua_pede_esclarecimento(self):
        ambito = Ambito(tema="dependente", seguimento=True, usa_imagem=False, consulta="")
        with patch("src.rag.decidir_ambito", return_value=ambito), patch("src.rag.recuperar") as pesquisa:
            resposta = responder("E essa doença?", "vlm_rag", historico_com_correcao())
        self.assertEqual(resposta.estado, "esclarecimento")
        self.assertEqual(resposta.motivo, "referencia_ambigua")
        pesquisa.assert_not_called()

    def test_nova_fotografia_nao_herda_correcao_de_outro_caso(self):
        historico = historico_com_correcao()
        historico.append(historico[0])
        consulta = "Como prevenir esca?"
        ambito = Ambito(tema="viticultura", seguimento=True, usa_imagem=False, consulta=consulta)
        with patch("src.rag.decidir_ambito", return_value=ambito), patch("src.rag.recuperar", return_value=[]) as pesquisa:
            responder("Como prevenir essa doença?", "vlm_rag", historico)
        self.assertEqual(pesquisa.call_args.args[1], consulta)

    def test_mudanca_de_tema_continua_a_ser_rejeitada(self):
        fora = Ambito(tema="fora", seguimento=False, usa_imagem=False, consulta="Receita de lasanha")
        with patch("src.rag.conversar", return_value=fora) as modelo:
            resultado = decidir_ambito(self.config, "Dá-me uma receita de lasanha.", historico_com_correcao(), False, None)
        self.assertEqual(resultado.tema, "fora")
        modelo.assert_called_once()

    def test_falta_de_fontes_continua_a_causar_recusa(self):
        ambito = Ambito(tema="viticultura", seguimento=True, usa_imagem=False, consulta="Como tratar black rot?")
        with patch("src.rag.decidir_ambito", return_value=ambito), patch("src.rag.recuperar", return_value=[]):
            resposta = responder("Como tratar?", "vlm_rag", historico_com_correcao())
        self.assertEqual(resposta.estado, "recusa")
        self.assertEqual(resposta.motivo, "similaridade")

    def test_afirmacao_sem_suporte_continua_a_ser_rejeitada(self):
        ambito = Ambito(tema="viticultura", seguimento=True, usa_imagem=False, consulta="Como tratar black rot?")
        with patch("src.rag.decidir_ambito", return_value=ambito), \
                patch("src.rag.recuperar", return_value=[PASSAGEM]), \
                patch("src.rag.gerar_partes", return_value=([Parte("Afirmação não sustentada.", [1])], "")), \
                patch("src.rag.confirmar_suporte", return_value=False):
            resposta = responder("Como tratar?", "vlm_rag", historico_com_correcao())
        self.assertEqual(resposta.estado, "recusa")
        self.assertEqual(resposta.motivo, "evidencia_insuficiente")

    def test_hipotese_visual_e_separada_das_citacoes_documentais(self):
        ambito = Ambito(tema="viticultura", seguimento=False, usa_imagem=True, consulta="Sinais compatíveis com esca")
        visual = Visual(motor="VLM", estado="ok", classe="esca", confianca=0.85, descricao="Manchas na folha.")
        passagem = {**PASSAGEM, "titulo": "Documento de ensaio sobre esca", "texto": "O documento descreve sintomas de esca."}
        with patch("src.rag.decidir_ambito", return_value=ambito), \
                patch("src.rag.analisar_vlm", return_value=visual), \
                patch("src.rag.recuperar", return_value=[passagem]), \
                patch("src.rag.gerar_partes", return_value=([Parte("Informação documental de ensaio.", [1])], "")) as gerar, \
                patch("src.rag.confirmar_suporte", return_value=True):
            resposta = responder("Que doença tem esta folha?", "vlm_rag", imagem=Image.new("RGB", (20, 20)))
        self.assertIn("Hipótese visual: Esca", resposta.partes[0]["texto"])
        self.assertEqual(resposta.partes[0]["fontes"], [])
        self.assertEqual(resposta.partes[1]["fontes"], ["D06:2"])
        self.assertNotIn("Manchas na folha.", gerar.call_args.args[1][-1].content)

    def test_recusa_visual_explica_limite_da_identificacao(self):
        ambito = Ambito(tema="viticultura", seguimento=False, usa_imagem=True, consulta="Que doença tem esta folha?")
        visual = Visual(motor="VLM", estado="ok", classe="esca", confianca=0.85)
        with patch("src.rag.decidir_ambito", return_value=ambito), \
                patch("src.rag.analisar_vlm", return_value=visual), \
                patch("src.rag.recuperar", return_value=[PASSAGEM]), \
                patch("src.rag.gerar_partes", return_value=([], "geracao")):
            resposta = responder("Que doença tem esta folha?", "vlm_rag", imagem=Image.new("RGB", (20, 20)))
        self.assertEqual(resposta.estado, "recusa")
        self.assertIn("Não consigo confirmar a doença", resposta.texto)
        self.assertEqual(resposta.fontes, [])


if __name__ == "__main__":
    unittest.main()
