"""Executar as experiências e guardar uma resposta JSON por item e repetição."""
import argparse
import csv
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

from .apoio import ErroProjeto, RAIZ, agora, gravar_json, ler_json, sha256, FICHEIRO_SPLITS, PASTA_IMAGENS
from .config import EXPERIENCIAS, MODOS, carregar_config
from .corpus import CSV_CORPUS, ler_corpus, selecionadas
from .imagem import CLASSES, analisar_cnn, config_cnn, ler_imagem
from .indice import descrever_colecao
from .modelos import ErroFormato, digest_modelo
from .preparar_cnn import validar_manifesto
from .rag import atos_mencionados, responder
from .vlm import analisar_vlm

PERGUNTAS = RAIZ / "data/perguntas/perguntas.csv"
RUNS = RAIZ / "runs"
CONDICOES_TEXTO = ("llm_cnn", "llm_rag_sem_limiar", "llm_cnn_rag")


def ler_perguntas():
    with PERGUNTAS.open(encoding="utf-8-sig", newline="") as ficheiro:
        dados = list(csv.DictReader(ficheiro))
    fontes = ler_corpus()
    indice = selecionadas(fontes)
    ids = set()
    for linha in dados:
        identificador = linha["id"]
        doc = linha["doc_id"]
        if not re.fullmatch(r"Q\d+", identificador) or identificador in ids:
            raise ErroProjeto(f"Identificador de pergunta inválido ou duplicado: {identificador}")
        ids.add(identificador)
        if linha["split"] not in {"validacao", "teste"} or not linha["pergunta"].strip():
            raise ErroProjeto(f"{identificador}: split ou pergunta inválida.")
        acao = linha["expected_action"]
        if acao not in {"responder", "recusar"}:
            raise ErroProjeto(f"{identificador}: expected_action inválido.")
        if doc not in fontes or (doc in indice) != (acao == "responder"):
            raise ErroProjeto(f"{identificador}: fonte e ação esperada são incompatíveis.")
        if not 1 <= int(linha["pagina_pdf"]) <= int(fontes[doc]["paginas"]):
            raise ErroProjeto(f"{identificador}: página fora dos limites do PDF.")
    return dados


def commit_atual():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def caminho_execucao(nome):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,70}", nome) or nome in {"chats", "checkpoints"}:
        raise ErroProjeto("Utilizar letras, números, hífen ou underscore no nome da execução.")
    return RUNS / nome


def preparar_execucao(nome, parametros):
    pasta = caminho_execucao(nome)
    manifesto = pasta / "execucao.json"
    parametros = {
        **parametros, "commit": commit_atual(),
        "protocolo_sha256": sha256(RAIZ / "PROTOCOLO.md"),
        "ambiente": {"python": platform.python_version(), "plataforma": platform.platform()},
        "codigo_sha256": {p.name: sha256(p) for p in sorted((RAIZ / "src").glob("*.py"))},
    }
    if manifesto.exists():
        if ler_json(manifesto)["parametros"] != parametros:
            raise ErroProjeto("Esta execução já existe com outros parâmetros. Utilizar outro nome.")
    else:
        gravar_json(manifesto, {"execucao": nome, "criado": agora(), "parametros": parametros})
    return pasta


def gravar_item(pasta, chave, dados, calcular):
    destino = pasta / "itens" / f"{chave}.json"
    if destino.exists():
        return
    inicio = perf_counter()
    try:
        resultado = calcular()
    except ErroFormato as erro:
        resultado = {"estado": "erro_formato", "erro": str(erro)}
    except ErroProjeto as erro:
        gravar_json(pasta / "erros" / f"{chave}.json", {**dados, "erro": str(erro), "data": agora()})
        raise ErroProjeto(f"{chave}: {erro} As respostas já guardadas podem ser retomadas.") from erro
    item = {**dados, **resultado, "chave": chave, "data": agora(), "latencia_s": perf_counter() - inicio}
    gravar_json(destino, item)
    (pasta / "erros" / f"{chave}.json").unlink(missing_ok=True)
    print(f"{chave} · {item['estado']} · {item['latencia_s']:.1f} s", flush=True)


def avaliar_texto(args):
    linhas = [p for p in ler_perguntas() if p["split"] == args.split]
    if args.limite:
        linhas = linhas[:args.limite]
    configs = {modo: carregar_config(modo) for modo in args.condicoes}
    pesos = {}
    indice_info = None
    for modo, cfg in configs.items():
        pesos[modo] = digest_modelo(cfg, cfg.modelo)
        if cfg.rag:
            if indice_info is None:
                try:
                    indice_info = descrever_colecao(cfg)
                except ErroProjeto:
                    raise
                except Exception:
                    raise ErroProjeto("Não foi possível validar o Qdrant. Executar src.indice info.") from None

    pasta = preparar_execucao(args.execucao, {
        "tipo": "texto", "split": args.split, "ids": [p["id"] for p in linhas],
        "repeticoes": args.repeticoes, "temperaturas": args.temperaturas,
        "itens_previstos": len(linhas) * len(configs) * len(args.temperaturas) * args.repeticoes,
        "configuracoes": {m: c.model_dump() for m, c in configs.items()}, "modelos_digest": pesos,
        "perguntas_sha256": sha256(PERGUNTAS), "corpus_sha256": sha256(CSV_CORPUS),
        "indice_qdrant": indice_info,
    })
    for modo, cfg in configs.items():
        for temp in args.temperaturas:
            for linha in linhas:
                for rep in range(1, args.repeticoes + 1):
                    config = cfg.model_copy(update={"temperatura": temp, "seed": cfg.seed + rep - 1})
                    chave = f"{linha['id']}_{modo}_t{temp:g}_r{rep}"
                    dados = {
                        "tipo": "texto", "split": args.split, "modo": modo,
                        "modelo": cfg.modelo, "temperatura": temp, "seed": config.seed,
                        "repeticao": rep, "referencia": linha, "top_k": cfg.top_k, "rag": cfg.rag,
                    }

                    def calcular():
                        # A referência fica no log. O assistente recebe apenas o texto da pergunta.
                        resultado = responder(linha["pergunta"], modo, config=config, registar=True)
                        return resultado.model_dump()

                    gravar_item(pasta, chave, dados, calcular)
    print(f"Execução guardada em {pasta}.")


def checkpoint_cnn(padrao, seed):
    caminho = Path(padrao.replace("{seed}", str(seed)))
    return caminho if caminho.is_absolute() else RAIZ / caminho


def avaliar_imagem(args):
    manifesto = ler_json(args.splits)
    seeds = args.seeds or manifesto["seeds"]
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(manifesto["seeds"]):
        raise ErroProjeto("As seeds escolhidas devem ser distintas e estar em splits.json.")
    validar_manifesto(manifesto, seeds)
    imagens = {}
    for seed in seeds:
        # Preserva a comparação CNN/VLM nas quatro classes de folhas; Outro é rejeição.
        itens = [i for i in manifesto["imagens"]
                 if i["splits"][str(seed)] == args.split and i["classe"] in CLASSES]
        imagens[seed] = itens[:args.limite] if args.limite else itens
    config = carregar_config("vlm")
    modelos = {}
    checkpoints = {}
    if "vlm" in args.modelos:
        modelos["vlm"] = digest_modelo(config, config.modelo)
    if "cnn" in args.modelos:
        cfg_cnn = config_cnn()
        for seed in seeds:
            if seed != cfg_cnn.seed_treino:
                raise ErroProjeto("A seed das divisões deve coincidir com a seed de treino do checkpoint CNN.")
            caminho = checkpoint_cnn(args.cnn_checkpoint or cfg_cnn.checkpoint, seed)
            if not caminho.is_file():
                raise ErroProjeto(f"Falta o checkpoint da CNN para a seed {seed}: {caminho}.")
            checkpoints[seed] = caminho
            if sha256(caminho) != manifesto.get("proveniencia", {}).get("checkpoint_sha256"):
                raise ErroProjeto("O checkpoint não corresponde ao modelo recebido e registado no manifesto.")
        modelos["cnn"] = {str(s): sha256(c) for s, c in checkpoints.items()}
    pasta = preparar_execucao(args.execucao, {
        "tipo": "imagem", "split": args.split, "seeds": seeds, "seeds_treino": seeds,
        "seeds_amostragem": None, "modelos_digest": modelos,
        "config_vlm": config.model_dump() if "vlm" in args.modelos else None,
        "config_cnn": cfg_cnn.model_dump(mode="json") if "cnn" in args.modelos else None,
        "splits_sha256": sha256(args.splits), "limite": args.limite,
        "itens_previstos": sum(map(len, imagens.values())) * len(args.modelos),
    })
    raiz_imagens = args.imagens.resolve()
    for motor in args.modelos:
        for seed in seeds:
            for item in imagens[seed]:
                foto = (raiz_imagens / item["id"]).resolve()
                if not foto.is_relative_to(raiz_imagens) or not foto.is_file() or sha256(foto) != item["sha256"]:
                    raise ErroProjeto(f"Ficheiro em falta ou alterado: {item['id']}. Verificar a pasta FieldVitis.")
                chave = f"{motor}_s{seed}_{item['sha256'][:12]}_{Path(item['id']).stem}"
                dados = {
                    "tipo": "imagem", "split": args.split, "modo": motor, "seed": seed,
                    "seed_treino_cnn": seed, "seed_geracao_vlm": config.seed if motor == "vlm" else None,
                    "imagem": item["id"], "classe_ref": item["classe"],
                    "modelo": config.modelo if motor == "vlm" else checkpoints[seed].name,
                }

                def calcular():
                    imagem = ler_imagem(foto)
                    chamadas = []
                    visual = (analisar_vlm(config, imagem, chamadas) if motor == "vlm"
                              else analisar_cnn(imagem, checkpoints[seed]))
                    return {"estado": visual.estado, "visual": visual.model_dump(), "chamadas": chamadas}

                gravar_item(pasta, chave, dados, calcular)
    print(f"Execução guardada em {pasta}.")


def mostrar_dados():
    perguntas = ler_perguntas()
    for split in ("validacao", "teste"):
        rec = sum(p["split"] == split and p["expected_action"] == "recusar" for p in perguntas)
        resp = sum(p["split"] == split and p["expected_action"] == "responder" for p in perguntas)
        print(f"{split}: {resp} para responder, {rec} para recusar.")
    corpus = ler_corpus()
    fontes = selecionadas(corpus)
    print(f"Documentos para o índice: {', '.join(fontes)}")
    condicionais = [doc for doc, f in fontes.items() if f["indexar"] == "condicional"]
    print(f"Fontes condicionais cuja confirmação deve ser registada pelo grupo: {', '.join(condicionais)}")
    recusas = [p for p in perguntas if p["expected_action"] == "recusar"]
    disponiveis = atos_mencionados("\n".join(f["titulo"] for f in fontes.values()))
    abrangidas = [p["id"] for p in recusas if atos_mencionados(p["pergunta"]) - disponiveis]
    print(f"Recusas abrangidas pela regra de atos ausentes: {len(abrangidas)}/{len(recusas)} {abrangidas}.")
    print("Cobertura estática da regra de metadados; não são recusas observadas nem uma medida de recuperação.")
    print(f"Tipos das perguntas de recusa: {dict(Counter(p['tipo'] for p in recusas))}")
    manifesto = ler_json(FICHEIRO_SPLITS)
    for seed in manifesto["seeds"]:
        for split in ("treino", "validacao", "teste", "holdout"):
            classes = Counter(i["classe"] for i in manifesto["imagens"] if i["splits"][str(seed)] == split)
            print(f"CNN seed {seed}, {split}: {sum(classes.values())} imagens; classes {dict(classes)}.")


def avaliar_respostas_imagem(args):
    """Comparar respostas dos modos escolhidos com a mesma foto e sem histórico."""
    imagem = ler_imagem(args.foto)
    configs = {m: carregar_config(m) for m in args.condicoes}
    cnn = config_cnn() if any(c.gerador == "llm" for c in configs.values()) else None
    if cnn:
        checkpoint = checkpoint_cnn(cnn.checkpoint, cnn.seed_treino)
        if not checkpoint.is_file():
            raise ErroProjeto(f"Falta o checkpoint da CNN: {checkpoint}. Consultar runs/checkpoints/README.md.")
    indice = None
    for config in configs.values():
        if config.rag and indice is None:
            try:
                indice = descrever_colecao(config)
            except ErroProjeto:
                raise
            except Exception:
                raise ErroProjeto("Não foi possível validar o Qdrant. Executar src.indice info.") from None
    pasta = preparar_execucao(args.execucao, {
        "tipo": "resposta_imagem", "pergunta": args.pergunta,
        "imagem_sha256": sha256(args.foto), "imagem": args.foto.name,
        "configuracoes": {m: c.model_dump() for m, c in configs.items()},
        "modelos_digest": {m: digest_modelo(c, c.modelo) for m, c in configs.items()},
        "config_cnn": cnn.model_dump(mode="json") if cnn else None,
        "checkpoint_cnn_sha256": sha256(checkpoint) if cnn else None,
        "indice_qdrant": indice, "itens_previstos": len(configs),
    })
    for modo, config in configs.items():
        dados = {
            "tipo": "resposta_imagem", "modo": modo, "pergunta": args.pergunta,
            "imagem_sha256": sha256(args.foto), "historico": [],
            "imagem_na_geracao_final": config.gerador == "vlm",
        }
        def calcular():
            return responder(args.pergunta, modo, imagem=imagem, historico=(),
                             config=config, registar=True).model_dump()
        gravar_item(pasta, modo, dados, calcular)
    print(f"Respostas guardadas em {pasta}; comparar por avaliação humana.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="acao", required=True)
    texto = sub.add_parser("texto", help="Executar perguntas, sem mostrar referências ao modelo.")
    texto.add_argument("--condicoes", choices=EXPERIENCIAS, nargs="+", default=list(CONDICOES_TEXTO))
    texto.add_argument("--repeticoes", type=int, default=3)
    texto.add_argument("--temperaturas", nargs="+", type=float, default=[0.0, 0.7])
    imagem = sub.add_parser("imagem", help="Comparar CNN e VLM no mesmo split de imagens.")
    imagem.add_argument("--modelos", choices=["cnn", "vlm"], nargs="+", default=["vlm"])
    imagem.add_argument("--splits", type=Path, default=FICHEIRO_SPLITS)
    imagem.add_argument("--imagens", type=Path, default=PASTA_IMAGENS)
    imagem.add_argument("--seeds", type=int, nargs="+")
    imagem.add_argument("--cnn-checkpoint", help="Caminho dos pesos, com {seed} se houver um modelo por seed.")
    respostas = sub.add_parser("respostas-imagem", help="Comparar respostas dos quatro modos à mesma fotografia.")
    respostas.add_argument("--condicoes", choices=MODOS, nargs="+", default=["llm_cnn", "llm_cnn_rag", "vlm", "vlm_rag"])
    respostas.add_argument("--foto", type=Path, required=True)
    respostas.add_argument("--pergunta", required=True)
    respostas.add_argument("--execucao", required=True)
    for cmd in (texto, imagem):
        cmd.add_argument("--split", choices=["validacao", "teste", "holdout"] if cmd is imagem else ["validacao", "teste"], default="validacao")
        cmd.add_argument("--execucao", required=True, help="Nome da pasta em runs/. Repetir para retomar.")
        cmd.add_argument("--limite", type=int, help="Número de itens para um ensaio curto de validação.")
    sub.add_parser("dados", help="Validar IDs, páginas e divisão das perguntas.")
    args = parser.parse_args()
    if args.acao in {"texto", "imagem"} and args.limite is not None:
        if args.limite < 1 or args.split != "validacao":
            parser.error("--limite exige um número positivo e o split de validação.")
    if args.acao == "texto" and (
        args.repeticoes < 1 or any(not 0 <= t <= 2 for t in args.temperaturas)
        or len(set(args.temperaturas)) != len(args.temperaturas)
        or len(set(args.condicoes)) != len(args.condicoes)
    ):
        parser.error("Indicar repetições positivas, condições distintas e temperaturas distintas entre 0 e 2.")
    if args.acao == "imagem" and len(set(args.modelos)) != len(args.modelos):
        parser.error("Indicar modelos distintos.")
    if args.acao == "respostas-imagem" and len(set(args.condicoes)) != len(args.condicoes):
        parser.error("Indicar condições distintas.")
    try:
        if args.acao == "texto":
            avaliar_texto(args)
        elif args.acao == "imagem":
            avaliar_imagem(args)
        elif args.acao == "respostas-imagem":
            avaliar_respostas_imagem(args)
        else:
            mostrar_dados()
    except (ErroProjeto, OSError, ValueError) as erro:
        sys.exit(f"{erro}")


if __name__ == "__main__":
    main()
