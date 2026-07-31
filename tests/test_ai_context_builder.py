from ai.context_builder import AIContextBuilder
from vectorstore.vector_service import SearchResult


def make_result(
    chunk_id: str,
    document: str,
    file_path: str = "src/auth/AuthService.java",
    class_name: str = "AuthService",
    function_name: str = "authenticate",
    start_line: int = 10,
    end_line: int = 20,
    similarity_score: float = 0.9,
):
    return SearchResult(
        chunk_id=chunk_id,
        similarity_score=similarity_score,
        distance=0.0,
        file_name=file_path.split("/")[-1],
        file_path=file_path,
        language="java",
        class_name=class_name,
        function_name=function_name,
        start_line=start_line,
        end_line=end_line,
        document=document,
        repository_name="sample-repo",
        metadata={"repository_name": "sample-repo"},
    )


def test_context_builder_deduplicates_and_merges_overlapping_chunks():
    builder = AIContextBuilder(token_limit=1000, merge_line_gap=2)
    results = [
        make_result("chunk-1", "public void authenticate() {\n  validate();", end_line=14),
        make_result("chunk-1", "public void authenticate() {\n  validate();", end_line=14),
        make_result("chunk-2", "  validate();\n  issueToken();\n}", start_line=13, end_line=18),
    ]

    context = builder.build(
        question="Explain authentication",
        retrieved_chunks=results,
        repository_metadata={"repo_name": "sample-repo"},
    )

    assert context.repository.repository_name == "sample-repo"
    assert len(context.chunks) == 1
    assert context.chunks[0].start_line == 10
    assert context.chunks[0].end_line == 18
    assert "issueToken();" in context.chunks[0].content
    assert context.source_files == ("src/auth/AuthService.java",)


def test_context_builder_truncates_when_token_limit_is_small():
    builder = AIContextBuilder(token_limit=40)
    long_text = "line\n" * 200
    results = [make_result("chunk-1", long_text)]

    context = builder.build(
        question="Explain authentication",
        retrieved_chunks=results,
        repository_metadata={"repo_name": "sample-repo"},
    )

    assert len(context.chunks) == 1
    assert context.truncated is True
    assert context.warnings
    assert "[truncated]" in context.chunks[0].content


def test_context_builder_preserves_repository_metadata_and_symbols():
    builder = AIContextBuilder(token_limit=1000)
    results = [
        make_result(
            "chunk-1",
            "String generateToken(User user) { return jwt; }",
            file_path="src/security/JwtUtil.java",
            class_name="JwtUtil",
            function_name="generateToken",
            start_line=42,
            end_line=51,
        ),
    ]

    metadata = {
        "repo_name": "sample-repo",
        "repo_url": "https://github.com/example/sample-repo.git",
        "local_path": "repositories/sample-repo",
        "indexed_at": "2026-07-10T20:00:00",
        "stats": {"total_files": 128},
        "index_stats": {"chunks_generated": 980, "embeddings_stored": 980},
    }

    context = builder.build("Which file handles JWT?", results, metadata)

    assert context.repository.repo_url.endswith("sample-repo.git")
    assert context.repository.total_files == 128
    assert context.repository.total_chunks == 980
    assert context.repository.embeddings_stored == 980
    assert context.source_classes == ("JwtUtil",)
    assert context.source_functions == ("generateToken",)


def test_context_builder_keeps_negative_reranker_scores_with_default_threshold():
    builder = AIContextBuilder(token_limit=1000)
    results = [
        make_result(
            "negative-score",
            "def recognize_face(): return True",
            file_path="backend/recognition.py",
            class_name="",
            function_name="recognize_face",
            similarity_score=-1.25,
        )
    ]

    context = builder.build("Where is face recognition?", results, {"repo_name": "sample-repo"})

    assert len(context.chunks) == 1
    assert context.chunks[0].chunk_id == "negative-score"
