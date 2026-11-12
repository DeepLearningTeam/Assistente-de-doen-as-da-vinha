"""Aplicação Gradio. Executa com python -m app.app na raiz do projeto."""
import argparse
import html
import logging
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import gradio as gr

from src import chats
from src.apoio import ErroProjeto, RAIZ, agora
from src.config import MODOS, carregar_config
from src.imagem import CLASSES, LIMITE_BYTES, FotoInvalida, codificar_imagem, config_cnn, descodificar_imagem, ler_imagem
from src.rag import PERGUNTA_FOTO, responder

PAGINA = Path(__file__).resolve().parent
SEM_VISUAL = '<p class="painel-vazio">Anexa uma fotografia para ver a hipótese e os sinais observados.</p>'
SEM_FONTES = '<p class="painel-vazio">As referências da última resposta aparecem aqui.</p>'
DGAV = "https://www.dgav.pt/medicamentos/conteudo/produtos-fitofarmaceuticos/divulgacao/"
EXEMPLOS = {
    "Sintomas de míldio": "Como posso reconhecer sintomas de míldio nas folhas da videira?",
    "Doenças do lenho": "Por onde pode entrar na videira o fungo associado à eutipiose?",
}


def modelos_modo(modo, config=None):
    config = config or carregar_config(modo)
    modelos = {"resposta": config.modelo}
    if config.rag:
        modelos["embeddings"] = config.embeddings_modelo
    if config.gerador == "llm":
        cnn = config_cnn()
        modelos["cnn"] = cnn.checkpoint
        modelos["backend_cnn"] = cnn.backend
    return modelos


def html_modelos(modo):
    try:
        modelos = modelos_modo(modo)
        partes = [f"Resposta: <strong>{html.escape(modelos['resposta'])}</strong>"]
        if "embeddings" in modelos:
            partes.append(f"Embeddings RAG: {html.escape(modelos['embeddings'])}")
        if "cnn" in modelos:
            disponivel = (RAIZ / modelos["cnn"]).is_file()
            nota = "" if disponivel else " (ficheiro indisponível)"
            partes.append(f"CNN: {html.escape(modelos['cnn'])}{nota}")
        return '<div class="modelos-modo">' + "<br>".join(partes) + "</div>"
    except ErroProjeto as erro:
        return aviso_html(str(erro), erro=True)


def texto_md(texto):
    """Apresenta texto como texto, deixando os links a cargo da aplicação."""
    texto = html.escape(texto, quote=False)
    return re.sub(r"([\\`*_\[\]~|])", r"\\\1", texto)


def url_fonte(fonte):
    link = urlsplit(fonte["url"])
    if link.scheme not in {"https", "http"} or not link.netloc:
        return ""
    return urlunsplit(link._replace(fragment=f"page={fonte['pagina']}"))


def rotulo_fonte(fonte):
    return f"{fonte['doc']}, p. {fonte['pagina']}, {fonte['data']}"


def chat_gradio(chat):
    mensagens = []
    for turno in chat["turnos"]:
        pergunta = texto_md(turno["pergunta"])
        if turno.get("com_foto"):
            pergunta += "\n\n*Fotografia anexada*"
        mensagens.append({"role": "user", "content": pergunta})

        resposta = turno["resposta"]
        fontes = {f"{f['doc']}:{f['pagina']}": f for f in resposta["fontes"]}
        partes = []
        for parte in resposta["partes"]:
            texto = texto_md(parte["texto"])
            for chave in parte["fontes"]:
                fonte = fontes[chave]
                rotulo = rotulo_fonte(fonte)
                link = url_fonte(fonte)
                texto += f" [{rotulo}](<{link}>)" if link else f" [{rotulo}]"
            partes.append(texto)
        mensagem = "\n\n".join(partes) or texto_md(resposta["texto"])
        if resposta["estado"] == "resposta" and not fontes:
            mensagem += "\n\n*Resposta sem consulta ao corpus.*"
        modo = MODOS.get(turno["modo"], turno["modo"])
        modelo = turno.get("modelos", {}).get("resposta", "Modelo não registado")
        mensagem = f"*Modo: {texto_md(modo)} · Modelo: {texto_md(modelo)}*\n\n{mensagem}"
        mensagens.append({"role": "assistant", "content": mensagem})
    return mensagens


def html_visual(visual):
    if visual is None:
        return SEM_VISUAL
    estado = visual["estado"]
    if estado == "rejeitada":
        return f'<p class="erro-visual">{html.escape(visual["descricao"])}</p>'
    classe = CLASSES.get(visual["classe"], "Inconclusiva")
    if estado == "incerta" and visual["classe"]:
        classe += " · hipótese incerta"
    valor = visual["confianca"]
    confianca = ""
    if valor is not None:
        legenda = "Confiança declarada" if visual["motor"] == "VLM" else "Confiança da CNN"
        confianca = (
            f'<p class="confianca">{legenda} <strong>{valor:.0%}</strong></p>'
            f'<progress class="barra-visual" aria-label="{legenda}" value="{valor}" max="1"></progress>'
        )
    probabilidades = ""
    if visual["probabilidades"]:
        linhas = []
        for chave, p in sorted(visual["probabilidades"].items(), key=lambda item: -item[1]):
            label = CLASSES.get(chave, "Outra imagem")
            linhas.append(f"<li><span>{html.escape(label)}</span><strong>{p:.0%}</strong></li>")
        probabilidades = f'<ul class="distribuicao">{"".join(linhas)}</ul>'
    return (
        f'<p class="motor">Última fotografia · {visual["motor"]}</p>'
        f'<p class="hipotese">{html.escape(classe)}</p>{confianca}'
        f'<p class="descricao-visual">{html.escape(visual["descricao"])}</p>{probabilidades}'
    )


def html_fontes(resposta):
    if resposta is None:
        return SEM_FONTES
    fontes = resposta["fontes"]
    if not fontes:
        mensagem = "A última resposta não tem citações."
        if resposta["motivo"] in {"documento_ausente", "similaridade", "evidencia_insuficiente", "citacoes_invalidas", "geracao", "formato"}:
            mensagem = "A pesquisa não sustentou uma resposta citada."
        return f'<p class="painel-vazio">{mensagem}</p>'
    cartoes = []
    for fonte in fontes:
        rotulo = html.escape(rotulo_fonte(fonte))
        url = url_fonte(fonte)
        link = f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">Ver fonte ↗</a>' if url else ""
        antigo = fonte["data"][:4].isdigit() and date.today().year - int(fonte["data"][:4]) >= 5
        contexto = ["Brasil"] if fonte["contexto"] == "Brasil" else []
        if antigo:
            contexto.append("Fonte histórica")
        notas = f'<div class="notas-fonte">{html.escape(" · ".join(contexto))}</div>' if contexto else ""
        cartoes.append(
            f'<article class="fonte"><h3>{html.escape(fonte["titulo"])}</h3>'
            f'<p class="meta-fonte">{rotulo} · {html.escape(fonte["entidade"])}</p>'
            f'<p class="excerto">“{html.escape(fonte["excerto"])}”</p>{notas}{link}</article>'
        )
    return "".join(cartoes)


def aviso_html(mensagem, erro=False):
    if not mensagem:
        return '<div class="mensagem-estado" role="status"></div>'
    estilo = " estado-erro" if erro else ""
    return f'<div class="mensagem-estado{estilo}" role="status">{html.escape(mensagem)}</div>'


def criar_tema():
    """Mantém a paleta clara também quando o navegador prefere o modo escuro."""
    cores = {
        "body_background_fill": "#f3f5ed", "body_text_color": "#203527",
        "body_text_color_subdued": "#526558", "background_fill_primary": "#fffefa",
        "background_fill_secondary": "#eef3e9", "block_background_fill": "#fffefa",
        "block_border_color": "#d9e1d4", "border_color_primary": "#d9e1d4",
        "color_accent_soft": "#e3ecdf", "border_color_accent": "#497858",
        "border_color_accent_subdued": "#c8d8c2",
        "block_label_background_fill": "transparent", "block_label_border_color": "transparent",
        "block_label_text_color": "#526558", "block_title_background_fill": "transparent",
        "block_title_text_color": "#203527", "accordion_text_color": "#203527",
        "input_background_fill": "#ffffff", "input_background_fill_focus": "#ffffff",
        "input_background_fill_hover": "#ffffff", "input_border_color": "#d1dccd",
        "input_border_color_focus": "#497858", "input_border_color_hover": "#a8bba6",
        "input_placeholder_color": "#68796b", "button_primary_background_fill": "#294f36",
        "button_primary_background_fill_hover": "#21432d", "button_primary_border_color": "#294f36",
        "button_primary_border_color_hover": "#21432d", "button_primary_text_color": "#ffffff",
        "button_primary_text_color_hover": "#ffffff", "button_secondary_background_fill": "#edf2e9",
        "button_secondary_background_fill_hover": "#e3ecdf", "button_secondary_border_color": "#dce5d7",
        "button_secondary_border_color_hover": "#a8bba6", "button_secondary_text_color": "#284333",
        "button_secondary_text_color_hover": "#284333", "checkbox_label_text_color": "#203527",
        "link_text_color": "#2a5d3b", "link_text_color_hover": "#21432d",
    }
    tema = gr.themes.Soft(primary_hue="green", neutral_hue="stone", font=["Inter", "system-ui", "sans-serif"])
    return tema.set(**cores, **{f"{chave}_dark": valor for chave, valor in cores.items()})


def criar_app():
    with gr.Blocks(title="Assistente da Vinha", analytics_enabled=False, delete_cache=(86400, 86400)) as demo:
        estado = gr.State(value=chats.novo_chat)
        exemplos = []
        gr.HTML(
            """
            <header class="vinha-header">
              <div class="marca"><span class="marca-grafismo" aria-hidden="true"></span>
                <div><h1>Assistente da Vinha</h1>
                <p>Analisa uma folha de videira ou faz uma pergunta sobre a vinha.</p></div>
              </div>
              <span class="local"><span aria-hidden="true"></span> Execução local</span>
            </header>
            """,
            elem_id="cabecalho",
        )
        with gr.Row(elem_id="grelha"):
            with gr.Column(scale=7, min_width=330, elem_id="coluna-chat"):
                with gr.Column(elem_classes="cartao", elem_id="cartao-chat"):
                    gr.HTML('<h2 class="titulo-painel">Conversa</h2>')
                    conversa = gr.Chatbot(
                        show_label=False, height=350, container=False, elem_id="conversa",
                        buttons=["copy", "copy_all"], feedback_options=[], allow_tags=False,
                        placeholder=(
                            '<div class="chat-vazio"><strong>Como te posso ajudar com a vinha?</strong>'
                            '<p>Escreve uma pergunta ou envia uma fotografia para começar.</p></div>'
                        ),
                    )
                    with gr.Row(elem_id="exemplos"):
                        for rotulo, exemplo in EXEMPLOS.items():
                            botao = gr.Button(rotulo, size="sm", variant="secondary")
                            exemplos.append((botao, exemplo))
                    pergunta = gr.Textbox(
                        label="Pergunta", placeholder="Escreve a tua pergunta…",
                        lines=2, max_lines=6, elem_id="pergunta",
                    )
                    with gr.Row(elem_id="compositor"):
                        foto = gr.File(
                            label="Fotografia da folha (opcional)", file_types=[".jpg", ".jpeg", ".png"],
                            type="filepath", height=91, elem_id="foto-upload", scale=3,
                        )
                        with gr.Column(scale=2, min_width=205, elem_id="modo-enviar"):
                            modo = gr.Dropdown(
                                choices=[(label, chave) for chave, label in MODOS.items()],
                                value="llm_cnn_rag", label="Modo de análise", interactive=True,
                            )
                            modelos_ui = gr.HTML(html_modelos("llm_cnn_rag"), elem_id="modelos-modo")
                            enviar = gr.Button("Enviar", variant="primary", elem_id="enviar")
                    repetir_foto = gr.Checkbox(
                        label="Voltar a usar a última fotografia", visible=False, value=False,
                        elem_id="repetir-foto",
                    )
                    estado_ui = gr.HTML(aviso_html(""), elem_id="aviso")
            with gr.Column(scale=4, min_width=285, elem_id="coluna-lateral"):
                with gr.Accordion("Identificação visual", open=True, elem_classes="cartao", elem_id="painel-visual"):
                    visual = gr.HTML(SEM_VISUAL)
                    miniatura = gr.Image(
                        show_label=False, type="pil", interactive=False, visible=False, height=175,
                        buttons=["fullscreen"], alt_text="Última fotografia analisada", elem_id="miniatura",
                    )
                    mapa = gr.Image(
                        label="Mapa de atenção da CNN", interactive=False, visible=False, height=175,
                        buttons=["fullscreen"], alt_text="Mapa de atenção da CNN",
                    )
                with gr.Accordion("Fontes citadas", open=True, elem_classes="cartao", elem_id="painel-fontes"):
                    fontes = gr.HTML(SEM_FONTES)
                with gr.Column(elem_classes="cartao", elem_id="painel-chats"):
                    gr.HTML('<h2 class="titulo-painel">Histórico</h2>')
                    guardadas = gr.Dropdown(
                        label="Conversas guardadas", choices=chats.listar(),
                        value=None, interactive=True,
                    )
                    with gr.Row(elem_id="editar-nome"):
                        nome = gr.Textbox(
                            label="Nome da conversa", show_label=False, container=False,
                            placeholder="Nome da conversa", lines=1, min_width=0, scale=3,
                            elem_id="nome-conversa",
                        )
                        renomear = gr.Button("Renomear", size="sm", scale=1, min_width=85)
                    with gr.Row(elem_id="abrir-guardar"):
                        abrir_guardada = gr.Button("Abrir", size="sm")
                        guardar = gr.Button("Guardar", size="sm")
                    novo = gr.Button("Nova conversa", size="sm", elem_id="novo-chat")
        gr.HTML(
            f'<footer class="vinha-rodape"><strong>Apoio técnico experimental.</strong> '
            f'Confirma produtos e doses na <a href="{DGAV}" target="_blank" rel="noopener noreferrer">'
            'DGAV</a> ou com um técnico.</footer>',
            elem_id="rodape",
        )

        def mostrar(chat, aviso=""):
            turnos = chat["turnos"]
            ultima = turnos[-1]["resposta"] if turnos else None
            turno_visual = next((t for t in reversed(turnos) if t["resposta"].get("visual")), None)
            resultado = turno_visual["resposta"]["visual"] if turno_visual else None
            mini = turno_visual.get("foto") if turno_visual else None
            atencao = resultado.get("mapa_atencao") if resultado else None
            escolhas = chats.listar()
            selecionada = chat["id"] if any(identificador == chat["id"] for _, identificador in escolhas) else None
            return {
                estado: chat, conversa: chat_gradio(chat),
                nome: chat.get("titulo", ""),
                guardadas: gr.update(choices=escolhas, value=selecionada),
                visual: html_visual(resultado), fontes: html_fontes(ultima),
                miniatura: gr.update(visible=bool(mini), value=descodificar_imagem(mini) if mini else None),
                mapa: gr.update(visible=bool(atencao), value=descodificar_imagem(atencao) if atencao else None),
                repetir_foto: gr.update(visible=bool(chats.ultima_foto(chat)), value=False),
                pergunta: "", foto: None, estado_ui: aviso_html(aviso),
            }

        def enviar_turno(chat, texto, caminho_foto, modo_escolhido, reutilizar):
            yield {estado_ui: aviso_html("A preparar a resposta no Ollama…")}
            try:
                imagem_b64 = codificar_imagem(ler_imagem(caminho_foto)) if caminho_foto else None
                if not imagem_b64 and reutilizar:
                    imagem_b64 = chats.ultima_foto(chat)
                imagem = descodificar_imagem(imagem_b64) if imagem_b64 else None
                config = carregar_config(modo_escolhido)
                modelos = modelos_modo(modo_escolhido, config)
                resposta = responder(texto, modo_escolhido, chat["turnos"], imagem, config=config)
                dados = resposta.model_dump(exclude={"chamadas", "recuperadas", "consulta"})
                usa_foto = resposta.visual is not None and resposta.visual.estado != "rejeitada"
                turno = {
                    "pergunta": (texto or "").strip() or PERGUNTA_FOTO, "modo": modo_escolhido,
                    "com_foto": imagem is not None, "foto": imagem_b64 if usa_foto else None,
                    "resposta": dados,
                    "modelos": modelos, "criado": agora(),
                }
                chat = {
                    **chat,
                    "modo": modo_escolhido, "turnos": [*chat["turnos"], turno],
                }
                try:
                    chat = chats.guardar(chat)
                    aviso = "Conversa guardada automaticamente."
                except (ErroProjeto, OSError) as erro:
                    aviso = f"Resposta recebida, mas não foi possível guardar a conversa. {erro}"
                yield mostrar(chat, aviso)
            except FotoInvalida as erro:
                yield {estado_ui: aviso_html(str(erro), erro=True)}
            except ErroProjeto as erro:
                logging.warning("%s", erro, exc_info=erro.__cause__ is not None)
                yield {estado_ui: aviso_html(str(erro), erro=True)}

        def guardar_chat(chat, modo_escolhido):
            try:
                chat = chats.guardar({**chat, "modo": modo_escolhido})
                return {
                    estado: chat, nome: chat["titulo"],
                    estado_ui: aviso_html("Conversa guardada."),
                    guardadas: gr.update(choices=chats.listar(), value=chat["id"]),
                }
            except (ErroProjeto, OSError) as erro:
                return {estado_ui: aviso_html(f"Não foi possível guardar. {erro}", erro=True)}

        def selecionar_nome(identificador):
            try:
                return {nome: chats.titulo_chat(chats.abrir(identificador)) if identificador else ""}
            except ErroProjeto as erro:
                return {nome: "", estado_ui: aviso_html(str(erro), erro=True)}

        def aplicar_nome(chat, identificador, titulo):
            try:
                titulo = chats.normalizar_titulo(titulo)
                if not titulo:
                    raise ErroProjeto("Escreve o novo nome da conversa.")
                # Uma seleção no histórico pode ser diferente da conversa em curso.
                alvo = chats.abrir(identificador) if identificador and identificador != chat["id"] else chat
                alvo = {**alvo, "titulo": titulo}
                if alvo["turnos"]:
                    alvo = chats.guardar(alvo)
                resultado = {nome: titulo, estado_ui: aviso_html("Nome da conversa atualizado.")}
                if alvo["id"] == chat["id"]:
                    resultado[estado] = alvo
                resultado[guardadas] = gr.update(
                    choices=chats.listar(), value=alvo["id"] if alvo["turnos"] else None,
                )
                return resultado
            except (ErroProjeto, OSError) as erro:
                return {estado_ui: aviso_html(f"Não foi possível renomear. {erro}", erro=True)}

        def abrir_da_lateral(identificador):
            try:
                chat = chats.abrir(identificador)
                return {
                    **mostrar(chat, "Conversa aberta."), modo: chat["modo"],
                    modelos_ui: html_modelos(chat["modo"]),
                }
            except ErroProjeto as erro:
                return {estado_ui: aviso_html(str(erro), erro=True)}

        saidas = [estado, conversa, visual, fontes, miniatura, mapa, repetir_foto, pergunta, foto, estado_ui, nome, guardadas]
        saidas_gravacao = [estado, nome, guardadas, estado_ui]
        opcoes_evento = {"concurrency_id": "ollama_local", "api_visibility": "private", "show_progress": "hidden"}
        # O histórico Runs do Gradio só regista eventos com API pública.
        gr.on(
            [enviar.click, pergunta.submit], enviar_turno,
            [estado, pergunta, foto, modo, repetir_foto], saidas,
            **{**opcoes_evento, "api_visibility": "public"},
        )
        guardar.click(guardar_chat, [estado, modo], saidas_gravacao, **opcoes_evento)
        guardadas.input(selecionar_nome, guardadas, [nome, estado_ui], **opcoes_evento)
        gr.on([renomear.click, nome.submit], aplicar_nome, [estado, guardadas, nome], saidas_gravacao, **opcoes_evento)
        abrir_guardada.click(abrir_da_lateral, guardadas, [*saidas, modo, modelos_ui], **opcoes_evento)
        demo.load(lambda: gr.update(choices=chats.listar()), outputs=guardadas, **opcoes_evento)
        modo.change(html_modelos, modo, modelos_ui, queue=False, api_visibility="private")
        novo.click(
            lambda: {
                **mostrar(chats.novo_chat()), modo: "llm_cnn_rag", modelos_ui: html_modelos("llm_cnn_rag"),
            },
            outputs=[*saidas, modo, modelos_ui], **opcoes_evento,
        )
        for botao, exemplo in exemplos:
            # O exemplo preenche a caixa. O utilizador decide quando enviar.
            botao.click(lambda valor=exemplo: valor, outputs=pergunta, queue=False, api_visibility="private")
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--porta", type=int)
    args = parser.parse_args()
    tema = criar_tema()
    criar_app().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.porta, share=False,
        theme=tema, css_paths=PAGINA / "estilos.css", footer_links=["runs"], run_history=True,
        max_file_size=LIMITE_BYTES, enable_monitoring=False,
        blocked_paths=[str(RAIZ / "data"), str(RAIZ / "runs"), str(RAIZ / "configs")],
    )


if __name__ == "__main__":
    main()
