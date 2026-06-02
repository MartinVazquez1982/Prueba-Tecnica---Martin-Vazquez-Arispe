from pypdf import PdfReader
from pathlib import Path

def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n".join(pages)