from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from src.llm_baseline import criar_llm_local, responder_sem_rag


def main() -> None:
    parser = argparse.ArgumentParser(description='Teste rápido do LLM textual sem RAG e sem ferramentas.')
    parser.add_argument('--pergunta', default='O que é o míldio da videira e que sinais podem ser observados nas folhas?')
    parser.add_argument('--modelo', default=None)
    parser.add_argument('--temperatura', type=float, default=0.0)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    load_dotenv()
    llm = criar_llm_local(args.modelo, temperature=args.temperatura)
    resultado = responder_sem_rag(llm, args.pergunta, temperature=args.temperatura)
    registo = {
        'experimento': 'smoke_test_llm_sem_rag',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'pergunta': args.pergunta,
        'model_id': args.modelo,
        'temperature': args.temperatura,
        'tools': False,
        'rag': False,
        **resultado,
    }
    destino = Path(args.output) if args.output else Path('runs/llm_baseline_sem_rag') / (
        'smoke_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(registo, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(registo, ensure_ascii=False, indent=2))
    print(f'\nResultado guardado em: {destino}')


if __name__ == '__main__':
    main()
