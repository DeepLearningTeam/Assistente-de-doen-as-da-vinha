"""Guardar, consultar e exportar conversas como ficheiros JSON locais."""
import json
import re
from datetime import datetime
from uuid import uuid4

from .apoio import ErroProjeto, RAIZ, agora, gravar_json, ler_json
from .config import MODOS

PASTA_CHATS = RAIZ / "runs/chats"


def novo_chat():
    return {"id": uuid4().hex, "titulo": "", "criado": agora(), "modo": "llm_cnn_rag", "turnos": []}


def normalizar_titulo(nome):
    return " ".join((nome or "").split())


def titulo_chat(chat):
    if nome := normalizar_titulo(chat.get("titulo")):
        return nome
    if not chat["turnos"]:
        return "Nova conversa"
    primeira = " ".join(chat["turnos"][0]["pergunta"].split())
    return primeira[:57] + "…" if len(primeira) > 57 else primeira


def caminho_chat(identificador):
    if not identificador or not re.fullmatch(r"[0-9a-f]{32}", identificador):
        raise ErroProjeto("Seleciona uma conversa guardada.")
    return PASTA_CHATS / f"{identificador}.json"


def guardar(chat):
    if not chat["turnos"]:
        raise ErroProjeto("Escreve uma mensagem antes de guardar a conversa.")
    guardado = {**chat, "titulo": titulo_chat(chat), "guardado": agora()}
    gravar_json(caminho_chat(chat["id"]), guardado)
    return guardado


def abrir(identificador):
    try:
        return ler_json(caminho_chat(identificador))
    except (OSError, ValueError) as erro:
        raise ErroProjeto("Não foi possível abrir a conversa guardada.") from erro


def registos(pesquisa=""):
    """Resumo do histórico, sem transferir as fotografias para a tabela."""
    registos = []
    for caminho in PASTA_CHATS.glob("*.json"):
        try:
            chat = ler_json(caminho)
            if caminho.stem == chat["id"] and chat["turnos"]:
                data = datetime.fromisoformat(chat["guardado"]).astimezone().strftime("%d/%m/%Y %H:%M")
                modos = list(dict.fromkeys(MODOS.get(t["modo"], t["modo"]) for t in chat["turnos"]))
                modelos = list(dict.fromkeys(
                    t.get("modelos", {}).get("resposta", "Não registado") for t in chat["turnos"]
                ))
                titulo = titulo_chat(chat)
                texto = " ".join([
                    titulo, chat["id"], *modos, *modelos,
                    *(f"{t['pergunta']} {t['resposta']['texto']}" for t in chat["turnos"]),
                ])
                if pesquisa.casefold().strip() not in texto.casefold():
                    continue
                registos.append({
                    "id": chat["id"], "titulo": titulo, "data": data, "guardado": chat["guardado"],
                    "turnos": len(chat["turnos"]), "modos": ", ".join(modos), "modelos": ", ".join(modelos),
                })
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(registos, key=lambda r: (r["guardado"], r["id"]), reverse=True)


def listar():
    return [(f"{r['titulo']} · {r['data']}", r["id"]) for r in registos()]


def exportar_texto(chat):
    """Conversa integral para diagnóstico, com fontes e sem bytes de imagens."""
    if not chat["turnos"]:
        return ""
    linhas = [
        f"# {titulo_chat(chat)}", "", f"ID: {chat['id']}", f"Criada: {chat['criado']}",
        f"Última gravação: {chat.get('guardado', 'Ainda não guardada')}",
    ]
    for numero, turno in enumerate(chat["turnos"], 1):
        resposta = turno["resposta"]
        modo = turno["modo"]
        linhas += ["", f"## Turno {numero}", f"Modo: {MODOS.get(modo, modo)}"]
        modelos = turno.get("modelos")
        linhas.append("Modelos: " + (json.dumps(modelos, ensure_ascii=False) if modelos else "Não registados nesta conversa antiga"))
        if turno.get("criado"):
            linhas.append(f"Data: {turno['criado']}")
        linhas += ["", "Tu:", turno["pergunta"]]
        if turno.get("com_foto"):
            linhas.append("[Fotografia anexada; os bytes da imagem não são incluídos nesta cópia.]")
        linhas += ["", "Assistente:", resposta["texto"], "", f"Estado: {resposta['estado']}"]
        if resposta.get("motivo"):
            linhas.append(f"Motivo: {resposta['motivo']}")
        # Conservar a associação entre as afirmações e as fontes, além do texto completo.
        if resposta.get("partes"):
            linhas += ["", "Partes e citações:", json.dumps(resposta["partes"], ensure_ascii=False, indent=2)]
        if resposta.get("fontes"):
            linhas += ["", "Fontes:", json.dumps(resposta["fontes"], ensure_ascii=False, indent=2)]
        if resposta.get("visual"):
            visual = {k: v for k, v in resposta["visual"].items() if k != "mapa_atencao"}
            linhas += ["", "Análise visual:", json.dumps(visual, ensure_ascii=False, indent=2)]
    return "\n".join(linhas) + "\n"


def ultima_foto(chat):
    return next((t["foto"] for t in reversed(chat["turnos"]) if t.get("foto")), None)
