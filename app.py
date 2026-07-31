"""Main Streamlit application for RepoMind AI."""

import streamlit as st
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import STREAMLIT_PAGE_CONFIG, REPOSITORIES_DIR
from github_loader import GitHubRepositoryLoader, RepositoryManager
from evaluation.dashboard import render_retrieval_evaluation_page
from ui.repository_dashboard import render_repository_intelligence_page
from ui.components import (
    apply_global_styles,
    empty_state,
    metric_card,
    onboarding_card,
    page_header,
    progress_step_list,
    safe_error,
    section_header,
    status_badge,
)
from utils import get_logger, load_metadata, save_metadata

logger = get_logger(__name__)

# Configure Streamlit page
st.set_page_config(**STREAMLIT_PAGE_CONFIG)

apply_global_styles()


NAV_PAGES = [
    "Repository Intelligence",
    "Add Repository",
    "Index Repository",
    "Retrieval Evaluation",
    "Manage Repositories",
    "About",
]


INDEXING_STEPS = (
    "Repository cloned",
    "Parsing files",
    "Creating chunks",
    "Generating embeddings",
    "Updating Chroma",
    "Building BM25",
    "Complete",
)


def navigate_to(page: str, repository_name: str | None = None) -> None:
    """Move between Streamlit pages while preserving repository context."""
    st.session_state["active_page"] = page
    if repository_name:
        st.session_state["selected_repository_name"] = repository_name
        st.session_state["repository_intelligence_active_repo"] = repository_name
    st.rerun()


def primary_language_from_stats(stats: dict) -> str:
    extensions = stats.get("files_by_extension") or {}
    if not extensions:
        return "Unknown"
    extension = max(extensions.items(), key=lambda item: item[1])[0]
    return str(extension).lstrip(".").upper() or "Unknown"


def repository_is_indexed_metadata(repo_data: dict) -> bool:
    index_stats = repo_data.get("index_stats", {}) if isinstance(repo_data.get("index_stats"), dict) else {}
    return bool(repo_data.get("indexed_at")) and int(index_stats.get("embeddings_stored", 0) or 0) > 0

def main():
    """Main application function"""
    
    # Initialize session state
    if "current_repo" not in st.session_state:
        st.session_state.current_repo = None
    if "repos_list" not in st.session_state:
        st.session_state.repos_list = []
    if "active_page" not in st.session_state or st.session_state.active_page not in NAV_PAGES:
        st.session_state.active_page = "Repository Intelligence"
    
    # Navigation sidebar
    st.sidebar.markdown("## RepoMind AI")
    st.sidebar.caption("Repository Intelligence Platform")
    page = st.sidebar.radio(
        "Navigation",
        NAV_PAGES,
        index=NAV_PAGES.index(st.session_state.active_page),
        key="active_page",
        help="Choose a section to work with",
    )
    st.sidebar.divider()
    st.sidebar.caption("Understand unfamiliar codebases faster with hybrid retrieval and grounded AI insights.")
    
    if page == "Repository Intelligence":
        render_repository_intelligence_page()
    elif page == "Add Repository":
        show_repository_upload()
    elif page == "Index Repository":
        show_index_repository()
    elif page == "Retrieval Evaluation":
        render_retrieval_evaluation_page()
    elif page == "Manage Repositories":
        show_manage_repositories()
    else:
        show_about()


def show_repository_upload():
    """Repository upload and cloning page"""
    page_header(
        "Add Repository",
        "Clone a GitHub repository, review its source summary, then index it for retrieval and AI insights.",
    )
    section_header("Step 1: Paste GitHub Repository URL", "Use a public or accessible GitHub repository URL.")
    onboarding_card(
        "Start here",
        "Add a repository first. RepoMind will summarize what it found, then guide you to indexing so Ask RepoMind can answer with evidence.",
    )
    
    # Input form
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/username/repository.git",
            help="Full URL to GitHub repository"
        )
    
    with col2:
        st.write("")
        st.write("")
        clone_button = st.button("Clone Repository", key="clone_btn", type="primary", use_container_width=True)
    
    # Overwrite option
    overwrite = st.checkbox("Overwrite existing repository", value=False)
    
    if clone_button:
        if not repo_url.strip():
            st.error("Please enter a repository URL.")
            return
        
        with st.spinner("Cloning repository. This may take a moment..."):
            try:
                # Initialize loader and clone
                loader = GitHubRepositoryLoader(repo_url)
                success, message = loader.clone_repository(overwrite=overwrite)
                
                if success:
                    st.session_state.current_repo = loader
                    
                    # Verify and get info
                    is_valid, verify_msg = loader.verify_repository()
                    if is_valid:
                        repo_info = loader.get_repository_info()
                        st.session_state["selected_repository_name"] = repo_info["name"]
                        
                        # Display repository information
                        section_header("Repository Added", "Review what RepoMind found before indexing.")
                        
                        # Get and display file list
                        files = loader.get_file_list(relative=True)
                        stats = loader.get_repository_stats()
                        detected_language = primary_language_from_stats(stats)
                        existing_metadata = load_metadata(str(loader.local_path / ".rag_metadata.json")) or {}
                        indexed = repository_is_indexed_metadata(existing_metadata)
                        success_body = (
                            "Indexing is not required because this repository already has stored embeddings."
                            if indexed
                            else "Indexing is required because RepoMind needs embeddings and retrieval indexes before it can answer repository questions."
                        )
                        onboarding_card(
                            "Repository cloned successfully",
                            success_body,
                        )

                        info_cols = st.columns(4)
                        with info_cols[0]:
                            metric_card("Repository", repo_info["name"], "Ready for indexing")
                        with info_cols[1]:
                            metric_card("Detected Language", detected_language, "Based on file types")
                        with info_cols[2]:
                            metric_card("Repository Size", stats.get("total_size", "0 B"), "Source footprint")
                        with info_cols[3]:
                            metric_card("Files", len(files), "Supported source files")

                        detail_cols = st.columns(2)
                        with detail_cols[0]:
                            metric_card("Branch", repo_info["branch"])
                        with detail_cols[1]:
                            metric_card("Latest Commit", repo_info["commit"])

                        st.caption(f"Last modified: {repo_info['last_modified']}")
                        
                        # Display statistics
                        try:
                            display_repository_stats(loader)
                        except Exception as chart_error:
                            logger.warning("Repository statistics rendering failed: %s", chart_error, exc_info=True)
                            safe_error("Repository was cloned, but the optional summary visualization could not be rendered.", chart_error)
                        
                        # Save metadata
                        if loader.save_repository_metadata():
                            st.success("Repository metadata saved.")
                            if indexed:
                                st.info("Indexing is not required because this repository already has stored embeddings.")
                            else:
                                if st.button(
                                    "Index Repository",
                                    key=f"goto_index_after_clone_{repo_info['name']}",
                                    type="primary",
                                    use_container_width=True,
                                ):
                                    navigate_to("Index Repository", repo_info["name"])
                else:
                    st.error(f"Repository cloning failed. {message}")
            
            except ValueError as e:
                safe_error("Invalid repository URL. Check the GitHub URL and try again.", e)
            except Exception as e:
                safe_error("Repository cloning failed. Check the repository URL and network connection.", e)
                logger.error(f"Error cloning repository: {str(e)}")


def display_repository_stats(loader):
    """Display repository statistics"""
    section_header("Repository Statistics")
    
    stats = loader.get_repository_stats()
    
    if stats:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric_card("Total Files", stats.get("total_files", 0))
        with col2:
            metric_card("Repository Size", stats.get("total_size", "0 B"))
        with col3:
            metric_card("File Types", len(stats.get("files_by_extension", {})))
        
        # Files by extension
        if stats.get("files_by_extension"):
            st.markdown("#### Files by Extension")
            ext_data = stats["files_by_extension"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("File Type Distribution:")
                for ext, count in sorted(ext_data.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"- **{ext}**: {count} files")
            
            # Visualize with bar chart
            with col2:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                
                fig, ax = plt.subplots(figsize=(8, 4))
                extensions = list(ext_data.keys())
                counts = list(ext_data.values())
                
                ax.barh(extensions, counts, color='#1f77b4')
                ax.set_xlabel('Number of Files')
                ax.set_title('Files by Extension')
                ax.grid(axis='x', alpha=0.3)
                
                st.pyplot(fig, use_container_width=True)
        
        # Largest files
        if stats.get("largest_files"):
            st.markdown("#### Largest Files")
            for i, file_info in enumerate(stats["largest_files"], 1):
                st.write(f"{i}. `{file_info['path']}` - {file_info['size']}")


def show_index_repository():
    """Parse, embed, and index a cloned repository."""
    page_header(
        "Index Repository",
        "Scan, parse, chunk, embed, and store repository code for hybrid retrieval.",
    )

    manager = RepositoryManager()
    repositories = manager.get_cloned_repositories()

    if not repositories:
        empty_state("No repositories yet.", "Add a GitHub repository before indexing.", "Open Add Repository to begin.")
        if st.button("Add Repository", type="primary", use_container_width=True):
            navigate_to("Add Repository")
        return

    repo_options = {
        repo["repo_name"]: repo
        for repo in sorted(repositories, key=lambda item: item.get("repo_name", "").lower())
    }

    selected_repo_name = st.session_state.get("selected_repository_name")
    repo_names = list(repo_options.keys())
    selected_index = repo_names.index(selected_repo_name) if selected_repo_name in repo_names else 0
    selected_name = st.selectbox(
        "Repository",
        options=repo_names,
        index=selected_index,
        key="index_repository_selected_name",
        help="Choose a cloned repository to parse and store in ChromaDB.",
    )
    st.session_state["selected_repository_name"] = selected_name
    selected_repo = repo_options[selected_name]

    section_header("Repository Selection", "Choose the cloned repository to index or re-index.")
    display_index_repository_summary(selected_repo)

    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        reindex = st.checkbox("Re-index", value=False, help="Replace existing ChromaDB records for this repository.")
    with col2:
        st.caption("Pipeline: scan files, parse code, generate chunks, create embeddings, store vectors, build retrieval indexes.")

    index_button = st.button("Index Repository", type="primary", use_container_width=True)

    progress_bar = st.progress(0)
    status_container = st.empty()
    stats_container = st.container()

    if index_button:
        try:
            result = run_indexing_pipeline(
                selected_repo=selected_repo,
                reindex=reindex,
                progress_bar=progress_bar,
                status_container=status_container,
            )

            with stats_container:
                st.session_state["selected_repository_name"] = selected_name
                st.session_state["repository_intelligence_active_repo"] = selected_name
                display_index_completion_card(result)
                if st.button(
                    "Open Repository Intelligence",
                    key=f"open_intelligence_after_index_{selected_name}",
                    type="primary",
                    use_container_width=True,
                ):
                    navigate_to("Repository Intelligence", selected_name)
        except Exception as e:
            progress_bar.progress(0)
            logger.error(f"Indexing failed for {selected_name}: {str(e)}", exc_info=True)
            status_container.error("Indexing failed. Expand logs or check the application logs for technical details.")
            with st.expander("Technical details"):
                st.code(str(e), language="text")


def display_index_repository_summary(repo_data):
    """Show the selected repository's current metadata before indexing."""
    stats = repo_data.get("stats", {})
    index_stats = repo_data.get("index_stats", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Repository", repo_data.get("repo_name", "Unknown"))
    with col2:
        metric_card("Source Files", stats.get("total_files", 0))
    with col3:
        metric_card("Chunks", index_stats.get("chunks_generated", 0))
    with col4:
        metric_card("Embeddings", index_stats.get("embeddings_stored", 0))

    indexed_at = repo_data.get("indexed_at") or "Not indexed"
    st.caption(f"Indexed: {indexed_at}")
    if repository_is_indexed_metadata(repo_data):
        st.info("This repository is already indexed. Re-index only if the source has changed or you want to refresh stored vectors.")
    else:
        onboarding_card(
            "Next step: create the retrieval index",
            "Indexing parses files, creates chunks, stores embeddings, updates Chroma, and builds retrieval indexes for Ask RepoMind.",
        )


def run_indexing_pipeline(selected_repo, reindex, progress_bar, status_container):
    """Run parse, embedding, storage, and metadata update for one repository."""
    from parser.parser_manager import ParserManager
    from vectorstore.vector_service import VectorService

    repo_name = selected_repo["repo_name"]
    repo_path = selected_repo["local_path"]
    status_logs = []
    started_at = time.perf_counter()

    def log_status(message, progress, active_step):
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_logs.append(f"[{timestamp}] {message}")
        progress_bar.progress(progress)
        status_container.markdown(
            f"{progress_step_list(INDEXING_STEPS, active_step)}"
            f"<div class='info-card-caption'>{status_logs[-1]}</div>",
            unsafe_allow_html=True,
        )

    log_status(f"Repository cloned: {repo_name}", 5, 0)
    if not Path(repo_path).exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    log_status("Parsing files", 20, 1)
    parser = ParserManager(repo_path, repo_name)
    chunks, parse_report = parser.parse_repository()
    if parse_report.get("error"):
        raise RuntimeError(parse_report["error"])
    if not chunks:
        raise RuntimeError("Parser did not generate any chunks for this repository.")

    files_processed = parse_report.get("files_processed", 0)
    log_status(f"Creating chunks: {len(chunks)} chunks from {files_processed} files", 45, 2)

    log_status("Generating embeddings", 65, 3)
    vector_service = VectorService(repository_name=repo_name)
    index_result = vector_service.index_chunks(chunks, reindex=reindex)

    elapsed_seconds = time.perf_counter() - started_at
    log_status("Updating Chroma", 82, 4)
    log_status("Building BM25", 92, 5)

    result = {
        "repository_name": repo_name,
        "files_processed": files_processed,
        "chunks_generated": len(chunks),
        "embeddings_generated": index_result.embedded_chunks,
        "embeddings_stored": index_result.stored_chunks,
        "duplicates": index_result.duplicate_chunks,
        "skipped": index_result.skipped_chunks,
        "indexing_time_seconds": elapsed_seconds,
        "chunks_by_language": index_result.chunks_by_language,
        "average_chunk_size": index_result.average_chunk_size,
        "indexed_at": datetime.now().isoformat(),
    }

    update_repository_index_metadata(repo_path, result)
    log_status("Complete", 100, 6)
    return result


def update_repository_index_metadata(repo_path, index_stats):
    """Persist indexing status into the repository metadata file."""
    metadata_path = Path(repo_path) / ".rag_metadata.json"
    metadata = load_metadata(str(metadata_path)) or {}
    metadata["indexed_at"] = index_stats["indexed_at"]
    metadata["index_stats"] = index_stats
    if not save_metadata(metadata, str(metadata_path)):
        raise RuntimeError(f"Failed to update repository metadata: {metadata_path}")


def display_index_stats_dashboard(result):
    """Render final indexing metrics."""
    elapsed = f"{result['indexing_time_seconds']:.2f}s"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Files Processed", result["files_processed"])
    with col2:
        metric_card("Chunks Generated", result["chunks_generated"])
    with col3:
        metric_card("Vectors Stored", result["embeddings_stored"])
    with col4:
        metric_card("Indexing Time", elapsed)

    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.metric("Embeddings Generated", result["embeddings_generated"])
    with detail_cols[1]:
        st.metric("Duplicates", result["duplicates"])
    with detail_cols[2]:
        st.metric("Skipped", result["skipped"])

    if result.get("chunks_by_language"):
        st.markdown("#### Chunks by Language")
        st.bar_chart(result["chunks_by_language"])


def display_index_completion_card(result):
    """Render the successful indexing handoff into Repository Intelligence."""
    onboarding_card(
        "Repository indexed successfully",
        "RepoMind can now answer grounded questions using this repository's parsed code, embeddings, Chroma records, and retrieval indexes.",
    )
    display_index_stats_dashboard(result)


def show_manage_repositories():
    """Manage cloned repositories"""
    page_header("Manage Repositories", "Review cloned repositories, recover missing metadata, and manage indexing state.")
    
    manager = RepositoryManager()
    repos = manager.get_cloned_repositories()
    unmanaged = manager.get_unmanaged_git_repositories()
    
    if not repos:
        empty_state("No managed repositories yet.", "Add a repository or recover metadata for an existing cloned Git folder.")
    else:
        st.markdown(f"Found **{len(repos)}** managed repositories")
    
    for repo_data in repos:
        repo_name = repo_data["repo_name"]
        stats = repo_data.get("stats", {})
        index_stats = repo_data.get("index_stats", {})
        indexed = bool(repo_data.get("indexed_at") and index_stats.get("embeddings_stored", 0) > 0)
        primary_language = "Unknown"
        language_counts = index_stats.get("chunks_by_language") or stats.get("files_by_extension") or {}
        if language_counts:
            primary_language = max(language_counts.items(), key=lambda item: item[1])[0]
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([0.34, 0.18, 0.18, 0.3])
            
            with col1:
                st.markdown(f"**{repo_name}**")
                st.caption(repo_data.get("repo_url") or "Local repository")
                status_badge("Indexed" if indexed else "Not indexed")
            with col2:
                st.metric("Files", stats.get("total_files", "N/A"))
            with col3:
                st.metric("Primary", str(primary_language).upper())
            with col4:
                st.caption(f"Last indexed: {repo_data.get('indexed_at') or 'Not indexed'}")
                action_cols = st.columns(2)
                with action_cols[0]:
                    st.button("Open", key=f"open_{repo_name}", disabled=True, use_container_width=True, help="Use Repository Intelligence from the sidebar.")
                with action_cols[1]:
                    st.button("Index", key=f"index_{repo_name}", disabled=True, use_container_width=True, help="Use Index Repository from the sidebar.")
                confirm = st.checkbox("Confirm delete", key=f"confirm_delete_{repo_name}")
                if st.button("Delete", key=f"delete_{repo_name}", type="secondary", disabled=not confirm, use_container_width=True):
                    success, message = manager.delete_repository(repo_name)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            
    if unmanaged:
        section_header("Recover Existing Git Repositories", "Valid cloned repositories missing RepoMind metadata.")
        for repo in unmanaged:
            with st.container(border=True):
                col1, col2 = st.columns([0.75, 0.25])
                with col1:
                    st.markdown(f"**{repo['repo_name']}**")
                    st.caption(repo.get("repo_url") or repo.get("local_path"))
                with col2:
                    if st.button("Recover Metadata", key=f"recover_{repo['repo_name']}", use_container_width=True):
                        success, message = manager.recover_repository_metadata(repo["repo_name"])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)


def show_about():
    """About page"""
    page_header(
        "About RepoMind AI",
        "AI-powered repository intelligence built on parsing, hybrid retrieval, reranking, and grounded AI insights.",
    )
    
    st.markdown("""
    RepoMind AI helps developers understand unfamiliar codebases faster with repository ingestion,
    AST-aware parsing, semantic and lexical retrieval, reranking, evaluation, and grounded AI reports.

    ### Core Capabilities

    - GitHub repository ingestion and local repository management
    - AST-aware parsing and intelligent code chunking
    - Metadata extraction for files, classes, functions, and line ranges
    - SentenceTransformer embeddings stored in ChromaDB
    - BM25 lexical retrieval with fresh-start recovery
    - Hybrid retrieval with score fusion, folder weighting, and symbol boosting
    - CrossEncoder reranking for higher-quality top results
    - Retrieval evaluation dashboard with benchmark reports
    - Grounded AI repository insights with source citations

    ### Architecture Flow

    Repository -> Parse -> Chunk -> Embed -> Vector + BM25 Retrieval -> Fusion -> Reranking -> AI Intelligence

    ### Primary Workflow

    1. Add a GitHub repository.
    2. Index the repository.
    3. Open Repository Intelligence.
    4. Generate architecture, onboarding, module, and technology reports.
    5. Use Retrieval Evaluation to benchmark search quality.
    """)


if __name__ == "__main__":
    main()
