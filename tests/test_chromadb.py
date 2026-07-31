from pathlib import Path

from embedding.embedding_manager import EmbeddingManager
from embedding.embedding_service import EmbeddingService
from embedding.model_config import EmbeddingModelConfig
from parser.chunk_model import CodeChunk
from vectorstore.chroma_manager import ChromaManager, ChromaManagerConfig
from vectorstore.vector_service import VectorService


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "auth" in lowered or "jwt" in lowered or "login" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "database" in lowered or "config" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def build_chunks():
    return [
        CodeChunk(
            chunk_id="chunk-auth",
            file_name="auth_service.py",
            file_path="src/auth_service.py",
            language="python",
            repository_name="sample-repo",
            content="def validate_jwt(token):\n    return verify_signature(token)",
            chunk_type="function",
            function_name="validate_jwt",
            start_line=1,
            end_line=2,
        ),
        CodeChunk(
            chunk_id="chunk-db",
            file_name="database.py",
            file_path="src/database.py",
            language="python",
            repository_name="sample-repo",
            content="def build_database_config():\n    return {'host': 'localhost'}",
            chunk_type="function",
            function_name="build_database_config",
            start_line=1,
            end_line=2,
        ),
    ]


def build_vector_service(tmp_path):
    fake_model = FakeSentenceTransformer()
    embedding_service = EmbeddingService(
        config=EmbeddingModelConfig(cache_enabled=False, cache_dir=Path(tmp_path) / "cache"),
        model=fake_model,
    )
    embedding_manager = EmbeddingManager(embedding_service=embedding_service)
    chroma_manager = ChromaManager(ChromaManagerConfig(persist_directory=Path(tmp_path) / "chroma"))
    vector_service = VectorService(
        repository_name="sample-repo",
        chroma_manager=chroma_manager,
        embedding_manager=embedding_manager,
    )
    return vector_service, fake_model


def test_chromadb_indexes_chunks_and_blocks_duplicates(tmp_path):
    vector_service, fake_model = build_vector_service(tmp_path)
    chunks = build_chunks()

    result = vector_service.index_chunks(chunks)
    assert result.stored_chunks == 2
    assert result.duplicate_chunks == 0
    assert result.total_chunks == 2

    repeat = vector_service.index_chunks(chunks)
    assert repeat.stored_chunks == 0
    assert repeat.duplicate_chunks == 2
    assert vector_service.collection_manager.get_collection().count() == 2
    assert len(fake_model.calls) >= 1
    integrity = vector_service.collection_manager.inspect_integrity()
    assert integrity.total_vectors == 2
    assert integrity.duplicate_chunk_ids == []
    assert integrity.missing_metadata_ids == []
    assert integrity.empty_document_ids == []
    assert integrity.empty_embedding_ids == []


def test_chromadb_supports_reindex_and_retrieval(tmp_path):
    vector_service, _ = build_vector_service(tmp_path)
    chunks = build_chunks()

    vector_service.index_chunks(chunks)
    vector_service.index_chunks(chunks, reindex=True)

    results = vector_service.search("JWT validation", top_k=5)

    assert results
    assert results[0].chunk_id == "chunk-auth"
    assert results[0].function_name == "validate_jwt"
    assert results[0].similarity_score > 0.5
