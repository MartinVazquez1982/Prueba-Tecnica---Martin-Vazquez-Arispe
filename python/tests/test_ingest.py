import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from ingest import (
    clean_text,
    load_documents,
)
from readers.json_reader import read_json
from readers.md_reader import read_md
from readers.pdf_reader import read_pdf
from readers.txt_reader import read_txt 


# ---------------------------------------------------------------------------
# read_txt
# ---------------------------------------------------------------------------

def test_read_txt(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("hello world", encoding="utf-8")
    assert read_txt(f) == "hello world"


# ---------------------------------------------------------------------------
# read_md
# ---------------------------------------------------------------------------

def test_read_md_strips_headers(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("## Title\nsome text", encoding="utf-8")
    result = read_md(f)
    assert "##" not in result
    assert "Title" in result


def test_read_md_strips_bold(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("**bold text**", encoding="utf-8")
    result = read_md(f)
    assert "**" not in result
    assert "bold text" in result


def test_read_md_strips_inline_code(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("`some_code()`", encoding="utf-8")
    result = read_md(f)
    assert "`" not in result
    assert "some_code()" in result


def test_read_md_strips_links(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("[click here](https://example.com)", encoding="utf-8")
    result = read_md(f)
    assert "https://example.com" not in result
    assert "click here" in result


# ---------------------------------------------------------------------------
# read_pdf
# ---------------------------------------------------------------------------

def _make_page(text):
    page = MagicMock()
    page.extract_text.return_value = text
    return page


def test_read_pdf_single_page(tmp_path):
    f = tmp_path / "doc.pdf"
    f.touch()
    with patch("readers.pdf_reader.PdfReader") as MockReader:
        MockReader.return_value.pages = [_make_page("hello pdf")]
        assert read_pdf(f) == "hello pdf"


def test_read_pdf_multiple_pages(tmp_path):
    f = tmp_path / "doc.pdf"
    f.touch()
    with patch("readers.pdf_reader.PdfReader") as MockReader:
        MockReader.return_value.pages = [_make_page("page one"), _make_page("page two")]
        result = read_pdf(f)
    assert result == "page one\npage two"


def test_read_pdf_skips_empty_pages(tmp_path):
    f = tmp_path / "doc.pdf"
    f.touch()
    with patch("readers.pdf_reader.PdfReader") as MockReader:
        MockReader.return_value.pages = [_make_page("real content"), _make_page(None), _make_page("")]
        result = read_pdf(f)
    assert result == "real content"


def test_read_pdf_all_empty_pages_returns_empty_string(tmp_path):
    f = tmp_path / "doc.pdf"
    f.touch()
    with patch("readers.pdf_reader.PdfReader") as MockReader:
        MockReader.return_value.pages = [_make_page(None), _make_page("")]
        result = read_pdf(f)
    assert result == ""


# ---------------------------------------------------------------------------
# read_json
# ---------------------------------------------------------------------------

def test_read_json(tmp_path):
    data = {"status": "active", "count": 5}
    f = tmp_path / "data.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    result = read_json(f)
    assert "Status: active" in result
    assert "Count: 5" in result


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

def test_clean_text_tabs_become_spaces():
    result = clean_text("hello\tworld")
    assert "\t" not in result
    assert "hello world" in result


def test_clean_text_collapses_multiple_spaces():
    result = clean_text("too   many   spaces")
    assert "  " not in result


def test_clean_text_collapses_blank_lines():
    result = clean_text("line1\n\n\n\nline2")
    assert result == "line1\n\nline2"


def test_clean_text_strips_trailing_whitespace():
    result = clean_text("line  \nother  ")
    for line in result.splitlines():
        assert line == line.rstrip()


def test_clean_text_unicode_normalization():
    result = clean_text("ﬁle")
    assert result == "file"


def test_clean_text_removes_control_chars():
    result = clean_text("hello\x01world")
    assert "\x01" not in result
    assert "helloworld" in result


def test_clean_text_normalizes_crlf():
    result = clean_text("line1\r\nline2")
    assert "\r" not in result
    assert "line1\nline2" in result


# ---------------------------------------------------------------------------
# load_documents  (returns list[Document])
# ---------------------------------------------------------------------------

def test_load_documents_reads_txt_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello from txt", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "a.txt"
    assert "hello from txt" in docs[0].page_content


def test_load_documents_skips_unsupported_extensions(tmp_path):
    (tmp_path / "a.txt").write_text("valid", encoding="utf-8")
    (tmp_path / "b.xyz").write_text("ignored", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "a.txt"


def test_load_documents_empty_dir_returns_empty_list(tmp_path):
    assert load_documents(tmp_path) == []


def test_load_documents_multiple_files(tmp_path):
    (tmp_path / "a.txt").write_text("file a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("file b", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 2
    assert {d.metadata["source"] for d in docs} == {"a.txt", "b.txt"}


def test_load_documents_skips_file_on_read_error(tmp_path):
    (tmp_path / "bad.txt").write_text("content", encoding="utf-8")

    def failing_reader(p):
        raise OSError("disk error")

    with patch("ingest.READERS", {".txt": failing_reader}):
        docs = load_documents(tmp_path)
    assert docs == []


def test_load_documents_returns_cleaned_text(tmp_path):
    (tmp_path / "c.txt").write_text("hello\t\t world", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert "\t" not in docs[0].page_content
