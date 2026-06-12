"""
Main Streamlit Application for GitHub Repository Understanding Assistant
Phase 1: Repository Ingestion and Management
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import STREAMLIT_PAGE_CONFIG, REPOSITORIES_DIR
from github_loader import GitHubRepositoryLoader, RepositoryManager
from utils import get_logger

logger = get_logger(__name__)

# Configure Streamlit page
st.set_page_config(**STREAMLIT_PAGE_CONFIG)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding-top: 0rem;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    if "current_repo" not in st.session_state:
        st.session_state.current_repo = None
    if "repos_list" not in st.session_state:
        st.session_state.repos_list = []
    
    # Header
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("🤖 GitHub Repository Understanding Assistant")
        st.markdown("**Powered by RAG (Retrieval-Augmented Generation)**")
    
    with col2:
        st.image("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", width=80)
    
    # Navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Repository Upload", "Manage Repositories", "About"],
        help="Choose a section to work with"
    )
    
    if page == "Repository Upload":
        show_repository_upload()
    elif page == "Manage Repositories":
        show_manage_repositories()
    else:
        show_about()


def show_repository_upload():
    """Repository upload and cloning page"""
    st.header("📥 Repository Upload & Cloning")
    
    st.markdown("""
    **Instructions:**
    1. Enter a valid GitHub repository URL
    2. Click "Clone Repository" to download the code
    3. The system will analyze the repository structure
    4. View statistics and prepare for indexing
    """)
    
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
        clone_button = st.button("🔄 Clone Repository", key="clone_btn", use_container_width=True)
    
    # Overwrite option
    overwrite = st.checkbox("Overwrite existing repository", value=False)
    
    if clone_button:
        if not repo_url.strip():
            st.error("❌ Please enter a repository URL")
            return
        
        with st.spinner(f"🔄 Cloning repository... This may take a moment."):
            try:
                # Initialize loader and clone
                loader = GitHubRepositoryLoader(repo_url)
                success, message = loader.clone_repository(overwrite=overwrite)
                
                if success:
                    st.session_state.current_repo = loader
                    st.success(f"✅ {message}")
                    
                    # Verify and get info
                    is_valid, verify_msg = loader.verify_repository()
                    if is_valid:
                        repo_info = loader.get_repository_info()
                        
                        # Display repository information
                        st.markdown("### Repository Information")
                        
                        info_cols = st.columns(3)
                        with info_cols[0]:
                            st.metric("Repository Name", repo_info["name"])
                        with info_cols[1]:
                            st.metric("Current Branch", repo_info["branch"])
                        with info_cols[2]:
                            st.metric("Latest Commit", repo_info["commit"])
                        
                        st.info(f"📍 **Local Path:** `{repo_info['local_path']}`")
                        st.caption(f"Last modified: {repo_info['last_modified']}")
                        
                        # Get and display file list
                        files = loader.get_file_list(relative=True)
                        st.success(f"Found {len(files)} supported source files")
                        
                        # Display statistics
                        display_repository_stats(loader)
                        
                        # Save metadata
                        if loader.save_repository_metadata():
                            st.info("✅ Repository metadata saved")
                else:
                    st.error(f"❌ {message}")
            
            except ValueError as e:
                st.error(f"❌ Invalid Repository URL: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error cloning repository: {str(e)}")


def display_repository_stats(loader):
    """Display repository statistics"""
    st.markdown("### 📊 Repository Statistics")
    
    stats = loader.get_repository_stats()
    
    if stats:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Files", stats.get("total_files", 0))
        with col2:
            st.metric("Total Size", stats.get("total_size", "0 B"))
        with col3:
            st.metric("File Types", len(stats.get("files_by_extension", {})))
        
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
            st.markdown("#### 📈 Largest Files")
            for i, file_info in enumerate(stats["largest_files"], 1):
                st.write(f"{i}. `{file_info['path']}` - {file_info['size']}")


def show_manage_repositories():
    """Manage cloned repositories"""
    st.header("🗂️ Manage Repositories")
    
    manager = RepositoryManager()
    repos = manager.get_cloned_repositories()
    
    if not repos:
        st.info("No repositories cloned yet. Go to 'Repository Upload' to clone a repository.")
        return
    
    st.markdown(f"Found **{len(repos)}** cloned repositories")
    
    # Display repositories
    for repo_data in repos:
        with st.expander(f"📦 {repo_data['repo_name']}", expanded=False):
            col1, col2 = st.columns([0.8, 0.2])
            
            with col1:
                st.markdown(f"**URL:** {repo_data['repo_url']}")
                st.markdown(f"**Path:** `{repo_data['local_path']}`")
                st.markdown(f"**Files:** {repo_data['stats'].get('total_files', 'N/A')}")
                st.markdown(f"**Size:** {repo_data['stats'].get('total_size', 'N/A')}")
                st.markdown(f"**Indexed:** {repo_data.get('indexed_at', 'Not indexed')}")
            
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{repo_data['repo_name']}"):
                    success, message = manager.delete_repository(repo_data['repo_name'])
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)


def show_about():
    """About page"""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## GitHub Repository Understanding Assistant
    
    A production-quality RAG (Retrieval-Augmented Generation) application that helps you understand any GitHub repository through intelligent code analysis.
    
    ### Features
    
    ✨ **Phase 1: Repository Ingestion**
    - Clone GitHub repositories automatically
    - Parse multiple programming languages
    - Organize and manage local copies
    
    📚 **Phase 2: Code Parsing** (Coming Soon)
    - Intelligent code chunking (function/class/method level)
    - Support for Python, Java, JavaScript, TypeScript, and more
    - Extract metadata and relationships
    
    🧠 **Phase 3: Embeddings** (Coming Soon)
    - Generate semantic embeddings for code chunks
    - Store in ChromaDB vector database
    - Similarity-based retrieval
    
    🔍 **Phase 4: Retrieval** (Coming Soon)
    - Semantic search across codebase
    - Return relevant code chunks with context
    - Show similarity scores
    
    💬 **Phase 5: LLM Integration** (Coming Soon)
    - Ask natural language questions
    - Get AI-powered answers with code references
    - Support for OpenAI and Groq APIs
    
    🎨 **Phase 6: Streamlit Interface** (Coming Soon)
    - User-friendly chat interface
    - Source code visualization
    - Repository statistics and insights
    
    ### Tech Stack
    
    - **Backend:** Python, LangChain, ChromaDB, Sentence Transformers
    - **Frontend:** Streamlit
    - **LLM:** OpenAI GPT / Groq Mixtral
    - **Repository Management:** GitPython
    
    ### Supported Languages
    
    - Python (.py)
    - Java (.java)
    - JavaScript (.js)
    - TypeScript (.ts, .tsx)
    - Markdown (.md)
    - YAML (.yaml, .yml)
    - And more...
    
    ### Getting Started
    
    1. Go to **Repository Upload** tab
    2. Enter a GitHub repository URL
    3. Click **Clone Repository**
    4. View statistics and manage your repositories
    
    ### Project Roadmap
    
    - [x] Phase 1: Repository Ingestion
    - [ ] Phase 2: Code Parsing & Chunking
    - [ ] Phase 3: Embeddings & Vector DB
    - [ ] Phase 4: Retrieval Pipeline
    - [ ] Phase 5: LLM Integration
    - [ ] Phase 6: Advanced UI Features
    - [ ] Phase 7: Testing & Deployment
    
    ---
    
    **Created with ❤️ for developers by developers**
    """)


if __name__ == "__main__":
    main()
