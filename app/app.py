"""Esqueleto Gradio. Esta UI não está ligada aos modelos até as funções serem integradas."""
from __future__ import annotations
import gradio as gr


def processar(pergunta: str, imagem, sistema: str):
    if not pergunta or not pergunta.strip():
        return "Escreve uma pergunta.", []
    # Não produzir respostas fictícias. Ligar aqui o agente LangChain / fluxo VLM quando pronto.
    return (
        "**Backend ainda não ligado.** A interface está apenas estruturada. Antes de usar, "
        "ligar e testar a CNN (F2), o recuperador documental (F3) e o fluxo selecionado.\n\n"
        f"Sistema selecionado: `{sistema}`.",
        [],
    )


with gr.Blocks(title="Assistente de doenças da vinha — P2 Tema F") as demo:
    gr.Markdown("# Assistente de doenças da vinha\nProtótipo de interface — backend ainda não integrado.")
    pergunta = gr.Textbox(label="Pergunta", placeholder="Descreve a dúvida sobre a folha ou doença...")
    imagem = gr.Image(type="pil", label="Fotografia da folha (opcional)")
    sistema = gr.Radio(
        choices=["LLM + CNN + ferramentas + RAG", "VLM + RAG"],
        value="LLM + CNN + ferramentas + RAG",
        label="Abordagem experimental",
    )
    botao = gr.Button("Consultar")
    resposta = gr.Markdown(label="Resposta")
    fontes = gr.JSON(label="Fontes recuperadas (quando integrado)")
    botao.click(processar, inputs=[pergunta, imagem, sistema], outputs=[resposta, fontes])


if __name__ == "__main__":
    demo.launch()
