from src.vlm import _extrair_classe, _extrair_linha
from src.prompts import BASELINE_SYSTEM_PROMPT, construir_prompt_sistema


def test_vlm_parser_accepts_canonical_and_common_labels():
    assert _extrair_classe("CLASSE: esca\nDESCRICAO: manchas") == "esca"
    assert _extrair_classe("CLASSE: leaf blight") == "leaf_blight"
    assert _extrair_classe("CLASSE: saudável") == "saudavel"
    assert _extrair_classe("CLASSE: resposta incerta") == "inconclusivo"


def test_vlm_parser_extracts_description_and_limitations():
    text = "CLASSE: esca\nDESCRICAO: manchas nas folhas\nLIMITACOES: imagem insuficiente"
    assert _extrair_linha(text, "DESCRICAO") == "manchas nas folhas"
    assert _extrair_linha(text, "LIMITACOES") == "imagem insuficiente"


def test_baseline_and_rag_use_same_prompt_template():
    rag_prompt = construir_prompt_sistema("[S1] D02, p. 8: exemplo")
    assert "És um assistente de apoio técnico à viticultura" in BASELINE_SYSTEM_PROMPT
    assert "És um assistente de apoio técnico à viticultura" in rag_prompt
    assert "[SEM DOCUMENTOS RECUPERADOS" in BASELINE_SYSTEM_PROMPT
    assert "[S1] D02, p. 8: exemplo" in rag_prompt
