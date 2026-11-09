from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from src.llm_tools import criar_llm_local
from src.tools_teste import criar_agente_demo


def texto(conteudo):
    if isinstance(conteudo, str):
        return conteudo
    if isinstance(conteudo, list):
        return '\n'.join(str(x.get('text', x)) if isinstance(x, dict) else str(x) for x in conteudo)
    return str(conteudo)


def main() -> None:
    parser = argparse.ArgumentParser(description='Testa tool calling LangChain com ferramentas simuladas.')
    parser.add_argument('--modelo', default=None)
    parser.add_argument('--pergunta', default='Recebi uma folha com sintomas. Usa as duas ferramentas de demonstração e explica o que devo verificar.')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    load_dotenv()
    llm = criar_llm_local(args.modelo)
    agente = criar_agente_demo(llm)
    resultado = agente.invoke({'messages': [{'role': 'user', 'content': args.pergunta}]})
    mensagens = resultado.get('messages', [])

    chamadas, executadas, tool_results = [], [], []
    final = ''
    for msg in mensagens:
        for chamada in getattr(msg, 'tool_calls', []) or []:
            chamadas.append({'nome': chamada.get('name'), 'args': chamada.get('args', {})})
        if getattr(msg, 'type', '') == 'tool':
            nome = getattr(msg, 'name', 'ferramenta')
            executadas.append(nome)
            tool_results.append({'nome': nome, 'resultado': texto(getattr(msg, 'content', ''))})
    for msg in reversed(mensagens):
        if getattr(msg, 'type', '') == 'ai' and getattr(msg, 'content', None):
            final = texto(msg.content)
            break

    esperadas = {'classificar_folha_demo', 'pesquisar_documentacao_demo'}
    faltam = sorted(esperadas - set(executadas))
    registo = {
        'experimento': 'llm_tools_demo',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model_id': args.modelo,
        'pergunta': args.pergunta,
        'chamadas_solicitadas_pelo_modelo': chamadas,
        'ferramentas_executadas': executadas,
        'resultados_ferramentas': tool_results,
        'ferramentas_em_falta': faltam,
        'resposta_final': final,
        'aviso': 'As ferramentas devolvem dados simulados. Não usar como evidência nem como resultado experimental.',
    }
    destino = Path(args.output) if args.output else Path('runs/llm_tools_demo') / (
        'tools_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(registo, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(registo, ensure_ascii=False, indent=2))
    print(f'\nRegisto guardado em: {destino}')
    if faltam:
        print('AVISO: o modelo não executou todas as ferramentas. Repete o teste ou experimenta um modelo com melhor suporte de tool calling.')
    else:
        print('As duas ferramentas de demonstração foram executadas.')


if __name__ == '__main__':
    main()
