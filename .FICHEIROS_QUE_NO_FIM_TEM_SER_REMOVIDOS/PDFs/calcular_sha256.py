import hashlib
from pathlib import Path

ficheiro = Path(__file__).parent / ("./D12-Despacho_n_69-G-2026.pdf")

if not ficheiro.is_file():
    raise FileNotFoundError(f"Ficheiro não encontrado: {ficheiro}")

sha256 = hashlib.sha256(ficheiro.read_bytes()).hexdigest()

print(sha256)
