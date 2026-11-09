from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from src.vlm import carregar_vlm


def main() -> None:
    parser = argparse.ArgumentParser(description='Executa inferência zero-shot do VLM numa fotografia local.')
    parser.add_argument('--imagem', required=True, help='Caminho para fotografia de folha de videira.')
    parser.add_argument('--pergunta', default='Que doença provável sugere a imagem? Descreve os sinais visuais observáveis.')
    parser.add_argument('--modelo', default=None)
    parser.add_argument('--max-new-tokens', type=int, default=180)
    parser.add_argument('--output', default=None, help='JSON de saída; por omissão guarda em runs/vlm_zero_shot/.')
    args = parser.parse_args()

    modelo = carregar_vlm(args.modelo)
    resultado = modelo.analisar(args.imagem, args.pergunta, args.max_new_tokens)
    resultado.update({
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'pergunta': args.pergunta,
        'max_new_tokens': args.max_new_tokens,
        'modo': 'zero_shot_visual_sem_rag',
    })

    out = Path(args.output) if args.output else Path('runs/vlm_zero_shot') / (
        'vlm_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    print(f'\nResultado guardado em: {out}')


if __name__ == '__main__':
    main()
