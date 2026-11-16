from src.rag import responder

r2 = responder("Como tratar oídio na vinha?", "llm_cnn", registar=True)
print(r2.estado, "|", r2.texto, "| fontes:", len(r2.fontes))