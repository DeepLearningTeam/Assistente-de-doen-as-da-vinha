r"""Com o Gradio aberto na porta 7865: ensaio HTTP com CNN e Ollama reais.

Executar da raiz: .\.venv\Scripts\python.exe verificacao/cnn/testar_gradio_real.py
Não executa inferência sobre o conjunto reservado.
"""
import json
from pathlib import Path

import httpx
from gradio_client import Client, handle_file

RAIZ = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:7865/"


def main():
    with httpx.Client(trust_env=False, timeout=20) as http:
        paginas = {p: http.get(URL + p).status_code for p in ("", "config", "gradio_api/runs")}
    assert all(s == 200 for s in paginas.values())
    manifesto = json.loads((RAIZ / "data/splits.json").read_text(encoding="utf-8"))
    item = next(i for i in manifesto["imagens"] if i["classe"] == "esca" and i["splits"]["1"] == "validacao")
    foto = RAIZ / "data/imagens" / item["id"]
    pasta = RAIZ / "runs/chats"
    anteriores = set(pasta.glob("*.json"))
    client = Client(URL, verbose=False)
    client.predict("Que sinais devo observar nesta folha de videira?", handle_file(str(foto)),
                   "llm_cnn", False, api_name="/enviar_turno")
    novos = set(pasta.glob("*.json")) - anteriores
    assert len(novos) == 1, "O envio não guardou uma nova conversa."
    ficheiro = novos.pop()
    chat = json.loads(ficheiro.read_text(encoding="utf-8"))
    primeiro = chat["turnos"][-1]
    assert primeiro["resposta"]["visual"]["motor"] == "CNN"
    client.predict("Que é a esca da videira?", None, "llm_cnn", False, api_name="/enviar_turno")
    chat = json.loads(ficheiro.read_text(encoding="utf-8"))
    ultimo = chat["turnos"][-1]
    assert len(chat["turnos"]) == 2
    assert ultimo["com_foto"] is False and ultimo["resposta"]["visual"] is None
    # Guardar resultados úteis sem os bytes da fotografia ou caminhos da cache.
    relatorio = {"http": paginas, "modo": "llm_cnn", "imagem_validacao": item["id"],
                 "fotografia": primeiro["resposta"], "pergunta_sem_fotografia": ultimo["resposta"],
                 "persistencia": True, "nao_reutilizou_cnn_implicitamente": True}
    (RAIZ / "verificacao/cnn/gradio_real.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Gradio real: HTTP 200, upload, CNN/Ollama, persistência e pergunta posterior sem fotografia verificados.")


if __name__ == "__main__":
    main()
