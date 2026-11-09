import hashlib
from pathlib import Path

ficheiro = Path(__file__).parent / (
    "./D19-Orientacoes-Producao-Integrada-Viticultura.pdf"
)

if not ficheiro.is_file():
    raise FileNotFoundError(f"Ficheiro não encontrado: {ficheiro}")

sha256 = hashlib.sha256(ficheiro.read_bytes()).hexdigest()

print(sha256)
