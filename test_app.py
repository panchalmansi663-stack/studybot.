from app import chunk_text, wants_summary


def test_wants_summary_detects_keyword():
    assert wants_summary("please summarize this document") is True


def test_wants_summary_ignores_unrelated_question():
    assert wants_summary("what is docker") is False


def test_chunk_text_short_text_returns_one_chunk():
    text = "This is a short paragraph that easily fits in one chunk."
    chunks = chunk_text(text, chunk_size=1500, overlap=200)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_long_text_into_multiple_chunks():
    paragraph = "A" * 100
    long_text = "\n\n".join([paragraph] * 10)
    chunks = chunk_text(long_text, chunk_size=200, overlap=20)
    assert len(chunks) > 1


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("", chunk_size=1500, overlap=200) == []