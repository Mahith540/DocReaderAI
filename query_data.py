def query_rag(query_text: str):
    # lazy import to avoid heavy dependencies at import time (useful for tests)
    from query_database import query_rag as _query_rag

    return _query_rag(query_text)


__all__ = ["query_rag"]
