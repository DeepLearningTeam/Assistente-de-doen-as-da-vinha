"""Contrato da CNN e prevenção de fuga na avaliação; serviços externos simulados."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

from src import cnn_resnet, rag, verificar_cnn
from src.apoio import ErroProjeto, RAIZ, ler_json
from src.config import carregar_config
from src.imagem import analisar_cnn, config_cnn, descodificar_imagem
from src.preparar_cnn import divisoes_originais, validar_manifesto


class ContratoCNN(unittest.TestCase):
    def test_configuracao_limitada_ao_contrato_resnet(self):
        config = config_cnn()
        self.assertEqual(set(config.model_dump()), {"checkpoint", "seed_treino", "classes"})
        with self.assertRaises(ValueError):
            type(config).model_validate({**config.model_dump(), "opcao_desconhecida": True})

    @unittest.skipUnless(importlib.util.find_spec("keras"), "Requer requirements.txt")
    def test_preprocessamento_rgb_bgr_float32_sem_dividir_255(self):
        imagem = Image.new("RGB", (20, 30), (10, 20, 30))
        x = cnn_resnet.preparar_imagem(imagem)
        self.assertEqual(x.shape, (1, 224, 224, 3))
        self.assertEqual(x.dtype, np.float32)
        np.testing.assert_allclose(x[0, 112, 112], [30 - 103.939, 20 - 116.779, 10 - 123.68], atol=1e-5)
        # A imagem não quadrada conserva a proporção; os cantos ficam pretos antes da normalização.
        np.testing.assert_allclose(x[0, 0, 0], [-103.939, -116.779, -123.68], atol=1e-5)

    @unittest.skipUnless(importlib.util.find_spec("keras"), "Requer requirements.txt")
    def test_exif_e_conversao_rgb(self):
        imagem = Image.new("L", (20, 20), 30)
        for y in range(10):
            for x in range(20):
                imagem.putpixel((x, y), 210)
        imagem.getexif()[274] = 3  # rotação 180 graus
        tensor = cnn_resnet.preparar_imagem(imagem)
        np.testing.assert_allclose(tensor[0, 0, 0], [30-103.939, 30-116.779, 30-123.68], atol=1e-5)
        np.testing.assert_allclose(tensor[0, -1, -1], [210-103.939, 210-116.779, 210-123.68], atol=1e-5)

    def test_ordem_neuronios_e_probabilidades_sem_segunda_softmax(self):
        valores = np.array([[0.05, 0.1, 0.15, 0.65, 0.05]], dtype=np.float32)
        with patch.object(cnn_resnet, "carregar_modelo") as modelo, \
             patch.object(cnn_resnet, "preparar_imagem", return_value=np.zeros((1, 224, 224, 3))), \
             patch.object(cnn_resnet, "gradcam", return_value=np.zeros((7, 7))):
            modelo.return_value.return_value = valores
            resultado = cnn_resnet.prever(Image.new("RGB", (4, 4)), Path("modelo.keras"))
        probs = resultado["probabilidades"]
        self.assertEqual(list(probs), ["black_rot", "esca", "saudavel", "leaf_blight", "outro"])
        np.testing.assert_allclose(list(probs.values()), valores[0])
        self.assertEqual(max(probs, key=probs.get), "leaf_blight")
        self.assertEqual(config_cnn().classes, list(probs))
        self.assertTrue(resultado["recognized"])
        self.assertIsInstance(resultado["mapa_atencao"], Image.Image)

    def test_rejeicao_nao_transporta_hipotese_de_doenca(self):
        probs = dict(zip(config_cnn().classes, [0.19, 0.2, 0.2, 0.22, 0.19]))
        with tempfile.NamedTemporaryFile() as f, \
             patch("src.imagem.prever", return_value={"probabilidades": probs, "recognized": False}):
            resultado = analisar_cnn(Image.new("RGB", (4, 4)), f.name)
        self.assertEqual(resultado.estado, "rejeitada")
        self.assertIsNone(resultado.classe)
        self.assertEqual(resultado.probabilidades, probs)

    def test_outro_e_limiar_fornecido_controlam_reconhecimento(self):
        casos = [
            ([0.1, 0.1, 0.1, 0.579, 0.121], False),
            ([0.1, 0.1, 0.1, 0.58, 0.12], True),
            ([0.05, 0.05, 0.05, 0.05, 0.8], False),
        ]
        for probs, esperado in casos:
            with self.subTest(probs=probs), patch.object(cnn_resnet, "carregar_modelo") as modelo, \
                    patch.object(cnn_resnet, "preparar_imagem"), \
                    patch.object(cnn_resnet, "gradcam", return_value=np.zeros((7, 7))) as mapa:
                modelo.return_value.return_value = np.array([probs])
                resultado = cnn_resnet.prever(Image.new("RGB", (4, 4)), Path("modelo.keras"))
            self.assertEqual(resultado["recognized"], esperado)
            self.assertEqual(mapa.call_count, int(esperado))
            if not esperado:
                self.assertIsNone(resultado["mapa_atencao"])

    def test_cache_carrega_apenas_uma_vez(self):
        cnn_resnet._carregar_modelo.cache_clear()
        self.addCleanup(cnn_resnet._carregar_modelo.cache_clear)
        with tempfile.NamedTemporaryFile() as f, patch.object(cnn_resnet, "obter_keras") as keras:
            modelo = keras.return_value.models.load_model.return_value
            modelo.input_shape, modelo.output_shape = (None, 224, 224, 3), (None, 5)
            modelo.layers[-1].get_config.return_value = {"activation": "softmax"}
            self.assertIs(cnn_resnet.carregar_modelo(Path(f.name)), cnn_resnet.carregar_modelo(Path(f.name)))
            keras.return_value.models.load_model.assert_called_once()

    def test_ficheiro_ausente_e_dependencia_ausente_dao_mensagem_util(self):
        with self.assertRaisesRegex(ErroProjeto, "Falta o modelo CNN"):
            cnn_resnet.carregar_modelo(Path("modelo_inexistente.keras"))
        with patch.dict("sys.modules", {"keras": None}), self.assertRaisesRegex(ErroProjeto, "requirements.txt"):
            cnn_resnet.obter_keras()

    def test_checkpoint_antigo_de_quatro_saidas_e_recusado(self):
        cnn_resnet._carregar_modelo.cache_clear()
        self.addCleanup(cnn_resnet._carregar_modelo.cache_clear)
        with tempfile.NamedTemporaryFile() as f, patch.object(cnn_resnet, "obter_keras") as keras:
            modelo = keras.return_value.models.load_model.return_value
            modelo.input_shape, modelo.output_shape = (None, 224, 224, 3), (None, 4)
            with self.assertRaisesRegex(ErroProjeto, "saída .*5"):
                cnn_resnet.carregar_modelo(Path(f.name))

    def test_probabilidades_invalidas_sao_rejeitadas(self):
        for valores in ([[float("nan"), 0, 0, 0, 1]], [[0.1] * 5], [[1, 2]]):
            with self.subTest(valores=valores), patch.object(cnn_resnet, "carregar_modelo") as m, \
                 patch.object(cnn_resnet, "preparar_imagem"), self.assertRaises(ErroProjeto):
                m.return_value.return_value = np.array(valores)
                cnn_resnet.prever(Image.new("RGB", (4, 4)), Path("modelo.keras"))

    def test_verificador_confere_saida_e_rejeicao_sem_codigo_de_origem(self):
        for aceite, valores in ((True, [0.05, 0.05, 0.05, 0.8, 0.05]),
                               (False, [0.05, 0.05, 0.05, 0.05, 0.8])):
            probabilidades = dict(zip(config_cnn().classes, valores))
            visual = rag.Visual(motor="CNN", estado="ok" if aceite else "rejeitada",
                                classe="leaf_blight" if aceite else None, confianca=0.8,
                                probabilidades=probabilidades)
            with self.subTest(aceite=aceite), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(verificar_cnn, "carregar_modelo") as carregar, \
                    patch.object(verificar_cnn, "imagens_validacao", return_value={"outro": {"id": "ensaio.png"}}), \
                    patch.object(verificar_cnn, "ler_imagem", return_value=Image.new("RGB", (8, 8))), \
                    patch.object(verificar_cnn, "preparar_imagem", return_value=np.zeros((1, 224, 224, 3), dtype=np.float32)), \
                    patch.object(verificar_cnn, "analisar_cnn", return_value=visual), \
                    patch.object(verificar_cnn, "version", return_value="simulada"), \
                    patch.object(verificar_cnn, "sha256", return_value="ensaio"):
                modelo = carregar.return_value
                modelo.input_shape, modelo.output_shape = (None, 224, 224, 3), (None, 5)
                modelo.predict.return_value = np.array([valores])
                modelo.count_params.return_value = 23597957
                camada = modelo.layers[-1]
                camada.name = "saida"
                camada.get_config.return_value = {"activation": "softmax"}
                modelo.layers = [camada]
                verificar_cnn.verificar_modelo(Path(tmp))
                relatorio = ler_json(Path(tmp) / "modelo_real.json")
                self.assertTrue(relatorio["inferencia_em_imagens"])
                self.assertTrue(relatorio["casos_validacao"][0]["rejeicao_consistente"])
                self.assertNotIn("paridade", json.dumps(relatorio))


class GradCAM(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("keras"), "Requer requirements.txt")
    def test_gradientes_por_classe_mapa_nulo_e_modelo_inalterado(self):
        keras = cnn_resnet.obter_keras()
        entrada = keras.Input((4, 4, 3))
        conv = keras.layers.Conv2D(2, 1, use_bias=False, kernel_initializer="zeros",
                                   name="conv5_block3_out")
        base = keras.Model(entrada, keras.layers.GlobalAveragePooling2D()(conv(entrada)), name="resnet50")
        foto = keras.Input((4, 4, 3))
        x = keras.Sequential([keras.layers.Activation("linear")])(foto)
        x = keras.layers.Dropout(0.3)(base(x))
        saida = keras.layers.Dense(5, activation="softmax", kernel_initializer="zeros")
        modelo = keras.Model(foto, saida(x))
        kernel = np.zeros((1, 1, 3, 2), dtype=np.float32)
        kernel[0, 0, 0, 0] = kernel[0, 0, 1, 1] = 1
        conv.set_weights([kernel])
        conv.trainable = False  # O Grad-CAM também deve funcionar com camadas congeladas.
        pesos = np.zeros((2, 5), dtype=np.float32)
        pesos[0, 0] = pesos[1, 1] = 1
        # Softmax saturada: o cálculo auxiliar deve usar o logit sem alterar a ativação.
        saida.set_weights([pesos, np.array([100, 0, 0, 0, 0], dtype=np.float32)])
        tensor = np.zeros((1, 4, 4, 3), dtype=np.float32)
        tensor[0, :, :2, 0] = tensor[0, :, 2:, 1] = 1
        previsao = np.asarray(modelo(tensor, training=False))
        configuracao = modelo.to_json()
        valores = [w.numpy().copy() for w in modelo.weights]
        camadas = [(l.trainable, len(l._inbound_nodes)) for l in modelo._flatten_layers()]
        ativacao = saida.activation
        esquerdo = cnn_resnet.gradcam(modelo, tensor, 0)
        direito = cnn_resnet.gradcam(modelo, tensor, 1)
        nulo = cnn_resnet.gradcam(modelo, tensor, 2)
        np.testing.assert_array_equal(esquerdo[:, :2], 1)
        np.testing.assert_array_equal(esquerdo[:, 2:], 0)
        np.testing.assert_array_equal(direito, 1 - esquerdo)
        self.assertTrue(np.isfinite(nulo).all())
        np.testing.assert_array_equal(nulo, 0)
        self.assertIs(saida.activation, ativacao)
        self.assertEqual(modelo.to_json(), configuracao)
        self.assertEqual([(l.trainable, len(l._inbound_nodes)) for l in modelo._flatten_layers()], camadas)
        for antes, depois in zip(valores, modelo.weights):
            np.testing.assert_array_equal(antes, depois.numpy())
        np.testing.assert_array_equal(modelo(tensor, training=False), previsao)

    def test_retira_padding_horizontal_e_vertical_antes_de_redimensionar(self):
        for tamanho in ((113, 224), (224, 113)):
            with self.subTest(tamanho=tamanho):
                imagem = Image.new("RGB", tamanho, (60, 80, 100))
                mapa = np.ones((224, 224), dtype=np.float32)
                # Estas margens ímpares reproduzem round(111 / 2) do ImageOps.pad.
                if tamanho[0] < 224:
                    mapa[:, 56:169] = 0
                else:
                    mapa[56:169, :] = 0
                sobreposicao = cnn_resnet.sobrepor_gradcam(imagem, mapa)
                self.assertEqual(sobreposicao.size, tamanho)
                self.assertEqual(sobreposicao.mode, "RGB")
                np.testing.assert_array_equal(sobreposicao, imagem)

    def test_mapa_nulo_preserva_fotografia_e_orientacao_EXIF(self):
        imagem = Image.new("RGB", (12, 8), "green")
        imagem.putpixel((0, 0), (255, 0, 0))
        imagem.getexif()[274] = 6
        resultado = cnn_resnet.sobrepor_gradcam(imagem, np.zeros((7, 7), dtype=np.float32))
        self.assertEqual(resultado.size, (8, 12))
        np.testing.assert_array_equal(resultado, ImageOps.exif_transpose(imagem).convert("RGB"))

    def test_mapa_PIL_chega_ao_campo_existente_da_interface(self):
        imagem = Image.new("RGB", (12, 8), "green")
        probs = dict(zip(config_cnn().classes, [0.05, 0.1, 0.1, 0.7, 0.05]))
        with tempfile.NamedTemporaryFile() as f, patch("src.imagem.prever", return_value={
            "probabilidades": probs, "recognized": True, "mapa_atencao": imagem,
        }):
            visual = analisar_cnn(imagem, f.name)
        self.assertEqual(visual.estado, "ok")
        self.assertIsNotNone(visual.mapa_atencao)
        self.assertEqual(descodificar_imagem(visual.mapa_atencao).size, imagem.size)


class DivisoesCNN(unittest.TestCase):
    def setUp(self):
        self.manifesto = ler_json(RAIZ / "data/splits.json")

    def test_divisoes_coincidem_com_treino_e_holdout_fica_separado(self):
        validar_manifesto(self.manifesto, [1])
        self.assertEqual({k: len(v) for k, v in divisoes_originais(1).items()},
                         {"treino": 454, "validacao": 98, "teste": 99, "holdout": 64})
        self.assertEqual(self.manifesto["seeds"], [1])

    def test_seed_sem_checkpoint_e_recusada(self):
        with self.assertRaisesRegex(ErroProjeto, "seed de treino 1"):
            validar_manifesto(self.manifesto, [0])

    def test_trocar_treino_por_teste_ou_repetir_imagem_falha(self):
        adulterado = copy.deepcopy(self.manifesto)
        item = next(i for i in adulterado["imagens"] if i["splits"]["1"] == "treino")
        item["splits"]["1"] = "teste"
        with self.assertRaisesRegex(ErroProjeto, "não corresponde"):
            validar_manifesto(adulterado, [1])
        adulterado = copy.deepcopy(self.manifesto)
        adulterado["imagens"].append(adulterado["imagens"][0])
        with self.assertRaisesRegex(ErroProjeto, "duplicados"):
            validar_manifesto(adulterado, [1])


class LigacaoAssistenteCNN(unittest.TestCase):
    def setUp(self):
        self.imagem = Image.new("RGB", (10, 10))
        self.visual = rag.Visual(motor="CNN", estado="ok", classe="leaf_blight", confianca=0.7,
                                 probabilidades={"black_rot": 0.05, "esca": 0.1, "saudavel": 0.1,
                                                 "leaf_blight": 0.7, "outro": 0.05})

    def test_rejeicao_cnn_termina_antes_de_recuperar_ou_gerar(self):
        ambito = rag.Ambito(tema="viticultura", seguimento=False, usa_imagem=True, consulta="Que folha é esta?")
        rejeitada = self.visual.model_copy(update={"estado": "rejeitada", "classe": None})
        for modo in ("llm_cnn", "llm_cnn_rag"):
            with self.subTest(modo=modo), patch.object(rag, "decidir_ambito", return_value=ambito), \
                    patch.object(rag, "analisar_cnn", return_value=rejeitada), \
                    patch.object(rag, "recuperar") as recuperar, patch.object(rag, "gerar_partes") as gerar:
                resposta = rag.responder("Que folha é esta?", modo, imagem=self.imagem)
            self.assertEqual(resposta.estado, "imagem_rejeitada")
            self.assertFalse(resposta.fontes)
            recuperar.assert_not_called()
            gerar.assert_not_called()

    def test_cnn_chega_ao_llm_e_consulta_rag_mesmo_sem_flag_usa_imagem(self):
        ambito = rag.Ambito(tema="viticultura", seguimento=False, usa_imagem=False, consulta="Que sinais devo confirmar?")
        for modo in ("llm_cnn", "llm_cnn_rag"):
            cfg = carregar_config(modo)
            with self.subTest(modo=modo), patch.object(rag, "decidir_ambito", return_value=ambito), \
                 patch.object(rag, "analisar_cnn", return_value=self.visual), \
                 patch.object(rag, "recuperar", return_value=[]) as recuperar, \
                 patch.object(rag, "gerar_partes", return_value=([rag.Parte("É uma hipótese.", [])], "")) as gerar:
                r = rag.responder("Que sinais devo confirmar?", modo, imagem=self.imagem, config=cfg)
            if cfg.rag:
                self.assertIn("Mancha-das-folhas (leaf blight)", recuperar.call_args.args[1])
                self.assertNotIn("míldio", r.consulta.lower())
            else:
                texto = gerar.call_args.args[1][-1].content
                self.assertIn('"confianca_softmax": 0.7', texto)
                self.assertIn('"leaf_blight": 0.7', texto)
                recuperar.assert_not_called()

    def test_nova_foto_substitui_cnn_e_sem_foto_nao_reutiliza_resultado(self):
        anterior = self.visual.model_copy(update={"classe": "esca"})
        historico = [{"pergunta": "Folha anterior?", "modo": "llm_cnn_rag", "resposta": rag.Resposta(
            estado="resposta", texto="Hipótese de esca.", visual=anterior).model_dump()}]
        ambito = rag.Ambito(tema="viticultura", seguimento=True, usa_imagem=True, consulta="Sinais na folha?")
        for imagem in (None, self.imagem):
            with self.subTest(com_foto=imagem is not None), patch.object(rag, "decidir_ambito", return_value=ambito), \
                 patch.object(rag, "analisar_cnn", return_value=self.visual) as cnn, \
                 patch.object(rag, "recuperar", return_value=[]) as recuperar:
                r = rag.responder("E nesta?", "llm_cnn_rag", historico, imagem)
            self.assertNotIn("Esca", recuperar.call_args.args[1])
            if imagem is None:
                self.assertIsNone(r.visual)
                cnn.assert_not_called()
            else:
                self.assertEqual(r.visual.classe, "leaf_blight")
                self.assertIn("leaf blight", r.consulta)

    def test_vlm_e_texto_nao_dependem_da_cnn(self):
        ambito = rag.Ambito(tema="viticultura", seguimento=False, usa_imagem=True, consulta="Folha?")
        for modo in ("llm_cnn", "llm_cnn_rag", "vlm", "vlm_rag"):
            imagens = (None, self.imagem) if modo.startswith("vlm") else (None,)
            for imagem in imagens:
                with self.subTest(modo=modo, foto=imagem is not None), \
                     patch.object(rag, "decidir_ambito", return_value=ambito), \
                     patch.object(rag, "analisar_cnn", side_effect=AssertionError("CNN chamada")), \
                     patch.object(rag, "analisar_vlm", return_value=self.visual.model_copy(update={"motor": "VLM"})), \
                     patch.object(rag, "recuperar", return_value=[]), \
                     patch.object(rag, "gerar_partes", return_value=([rag.Parte("Resposta.", [])], "")):
                    rag.responder("Folha?", modo, imagem=imagem)


if __name__ == "__main__":
    unittest.main()
