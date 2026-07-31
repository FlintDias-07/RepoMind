from ai.citation_manager import CitationManager
from ai.context_builder import AIContextBuilder
from ai.conversation_memory import ConversationMemory
from ai.prompt_builder import PromptBuilder
from ai.response_formatter import ResponseFormatter
from vectorstore.vector_service import SearchResult


def make_result(
    chunk_id: str,
    document: str,
    file_path: str = "src/auth/AuthService.java",
    class_name: str = "AuthService",
    function_name: str = "authenticate",
    start_line: int = 10,
    end_line: int = 20,
):
    return SearchResult(
        chunk_id=chunk_id,
        similarity_score=0.9,
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


def build_context():
    builder = AIContextBuilder(token_limit=1000)
    return builder.build(
        question="Explain authentication",
        retrieved_chunks=[
            make_result("chunk-1", "public void authenticate() {\n  validate();\n}", end_line=14),
            make_result(
                "chunk-2",
                "String generateToken(User user) { return jwt; }",
                file_path="src/security/JwtUtil.java",
                class_name="JwtUtil",
                function_name="generateToken",
                start_line=42,
                end_line=51,
            ),
        ],
        repository_metadata={
            "repo_name": "sample-repo",
            "stats": {"total_files": 64},
            "index_stats": {"chunks_generated": 220, "embeddings_stored": 220},
        },
    )


def test_prompt_builder_enforces_grounded_answering():
    context = build_context()
    prompt = PromptBuilder().build(
        context=context,
        role="software_architect",
        follow_up_history=["Q: Explain login", "A: Login is handled in controller and auth service."],
    )

    assert "Use ONLY the supplied repository context." in prompt.system_prompt
    assert "I could not find enough evidence in the indexed repository." in prompt.system_prompt
    assert "Conversation History:" in prompt.user_prompt
    assert "src/auth/AuthService.java" in prompt.user_prompt
    assert "src/security/JwtUtil.java" in prompt.user_prompt


def test_citation_manager_extracts_unique_citations():
    context = build_context()
    manager = CitationManager()
    citations = manager.from_context(context)

    assert len(citations) == 2
    assert manager.summarize_files(citations) == [
        "src/auth/AuthService.java",
        "src/security/JwtUtil.java",
    ]
    assert "JwtUtil" in manager.summarize_classes(citations)
    assert "generateToken" in manager.summarize_functions(citations)


def test_response_formatter_builds_structured_output():
    context = build_context()
    citations = CitationManager().from_context(context)
    raw_answer = (
        "Authentication is handled by AuthService and token generation is delegated to JwtUtil.\n"
        "AuthService.authenticate validates credentials before JwtUtil.generateToken issues the JWT."
    )

    response = ResponseFormatter().format(raw_answer=raw_answer, context=context, citations=citations)

    assert response.summary.startswith("Authentication is handled")
    assert "AuthService.authenticate" in response.detailed_explanation
    assert response.confidence_level == "high"
    assert response.referenced_files == (
        "src/auth/AuthService.java",
        "src/security/JwtUtil.java",
    )
    assert "## Summary" in response.rendered_markdown
    assert "## Referenced Evidence" in response.rendered_markdown


def test_response_formatter_handles_inline_numbered_summary():
    response = ResponseFormatter().format(
        raw_answer="1. Summary: Authentication is handled by AuthService.\n2. Detailed Explanation\nAuthService validates credentials.",
        context=build_context(),
        citations=CitationManager().from_context(build_context()),
    )

    assert response.summary == "Authentication is handled by AuthService."
    assert "AuthService validates credentials." in response.detailed_explanation


def test_response_formatter_handles_multiline_numbered_summary():
    response = ResponseFormatter().format(
        raw_answer="1. Summary\nAuthentication is handled by AuthService.\n2. Detailed Explanation\nAuthService validates credentials.",
        context=build_context(),
        citations=CitationManager().from_context(build_context()),
    )

    assert response.summary == "Authentication is handled by AuthService."
    assert response.summary != "1. Summary"


def test_response_formatter_handles_markdown_summary_heading():
    response = ResponseFormatter().format(
        raw_answer="## Summary\nAuthentication is handled by AuthService.\n## Detailed Explanation\nAuthService validates credentials.",
        context=build_context(),
        citations=CitationManager().from_context(build_context()),
    )

    assert response.summary == "Authentication is handled by AuthService."
    assert "AuthService validates credentials." in response.detailed_explanation


def test_response_formatter_handles_plain_summary_heading():
    response = ResponseFormatter().format(
        raw_answer="Summary:\nAuthentication is handled by AuthService.\nDetails:\nAuthService validates credentials.",
        context=build_context(),
        citations=CitationManager().from_context(build_context()),
    )

    assert response.summary == "Authentication is handled by AuthService."
    assert "AuthService validates credentials." in response.detailed_explanation


def test_response_formatter_falls_back_to_first_meaningful_paragraph_without_summary_heading():
    raw_answer = (
        "# Face Recognition Implementation\n\n"
        "SmartAttendanceAI detects faces, creates embeddings, and compares them to known records.\n\n"
        "## Implementation Flow\n"
        "1. Load models.\n2. Detect faces.\n3. Match embeddings."
    )
    response = ResponseFormatter().format(
        raw_answer=raw_answer,
        context=build_context(),
        citations=CitationManager().from_context(build_context()),
    )

    assert response.summary.startswith("SmartAttendanceAI detects faces")
    assert "# Face Recognition Implementation" in response.detailed_explanation


def test_response_formatter_preserves_custom_question_details():
    raw_answer = (
        "1. Summary: Face recognition uses model loading, detection, embeddings, and matching.\n\n"
        "## Implementation Flow\n"
        "1. Model Initialization\nMTCNN and Facenet are loaded.\n"
        "2. Face Detection\nFaces are detected from incoming frames.\n\n"
        "## Key Components\n- ModelManager\n- demo_recognize_optimized()"
    )
    response = ResponseFormatter().format(
        raw_answer=raw_answer,
        context=build_context(),
        citations=CitationManager().from_context(build_context()),
    )

    assert response.summary.startswith("Face recognition uses")
    assert "Model Initialization" in response.detailed_explanation
    assert "demo_recognize_optimized" in response.detailed_explanation


def test_response_formatter_marks_no_explicit_information_as_insufficient_evidence():
    context = build_context()
    response = ResponseFormatter().format(
        raw_answer="The repository does not contain explicit information about a Kubernetes cluster.",
        context=context,
        citations=CitationManager().from_context(context),
    )

    assert response.confidence_level == "low"
    assert ResponseFormatter.INSUFFICIENT_EVIDENCE_MESSAGE in response.warnings


def test_conversation_memory_is_repository_specific_and_bounded():
    memory = ConversationMemory(max_turns=2)
    memory.append_turn("repo-a", "session-1", "Explain auth", "Auth is in AuthService.")
    memory.append_turn("repo-a", "session-1", "Now explain JWT", "JWT is in JwtUtil.")
    memory.append_turn("repo-a", "session-1", "What about login?", "Login is in LoginController.")
    memory.append_turn("repo-b", "session-1", "Explain DB", "DB config is in application.yml.")

    repo_a_history = memory.render_history("repo-a", "session-1")
    repo_b_history = memory.render_history("repo-b", "session-1")

    assert len(repo_a_history) == 4
    assert "Explain auth" not in "\n".join(repo_a_history)
    assert "Now explain JWT" in "\n".join(repo_a_history)
    assert "Explain DB" in "\n".join(repo_b_history)
