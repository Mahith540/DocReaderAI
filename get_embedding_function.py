def get_embedding_function():
    # lazy import to avoid importing heavy deps during test collection
    from embedding_functions import get_embedding_function as _get_ef

    return _get_ef()


__all__ = ["get_embedding_function"]
