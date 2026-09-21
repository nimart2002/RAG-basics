from pathlib import Path
from src.load import cargar_corpus
from src.chunk import trocear

documentos = cargar_corpus(Path("data"))

for size, overlap in [(400, 50), (800, 100)]:
    chunks = trocear(documentos, chunk_size=size, chunk_overlap=overlap)
    print(f"chunk_size={size:4d} overlap={overlap:3d} -> {len(chunks)} chunks")