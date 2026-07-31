"""High-level indexing and semantic search service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

from parser.chunk_model import CodeChunk
from utils import get_logger

from embedding.embedding_manager import EmbeddingManager, EmbeddedChunk

from .chroma_manager import ChromaManager
from .collection_manager import CollectionManager, CollectionStatistics
from retrieval.bm25_retriever import BM25Retriever
from retrieval.collection_router import CollectionRouter
from retrieval.file_classifier import FileCategory
from retrieval.hybrid_pipeline import HybridRetrievalPipeline
from retrieval.query_classifier import QueryClassifier
from retrieval.ranking_pipeline import RankingPipeline


@dataclass
class SearchResult:
    """Normalized search result returned from semantic retrieval."""

    chunk_id: str
    similarity_score: float
    distance: float
    file_name: str
    file_path: str
    language: str
    class_name: str
    function_name: str
    start_line: int
    end_line: int
    document: str
    collection_name: str = ""
    repository_name: str = ""
    original_vector_score: float = 0.0
    bm25_score: float = 0.0
    fusion_score: float = 0.0
    folder_weight: float = 1.0
    metadata_bonus: float = 0.0
    symbol_boost: float = 0.0
    cross_encoder_score: float = 0.0
    reranked_position: int = 0
    adjusted_score: float = 0.0
    ranking_reasons: List[str] = field(default_factory=list)
    query_intent: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        if self.class_name and self.function_name:
            return f"{self.class_name}.{self.function_name}()"
        if self.function_name:
            return f"{self.function_name}()"
        if self.class_name:
            return f"class {self.class_name}"
        return self.file_path


@dataclass
class IndexResult:
    """Summary of an indexing run."""

    repository_name: str
    total_chunks: int
    embedded_chunks: int
    stored_chunks: int
    skipped_chunks: int
    duplicate_chunks: int
    total_files_processed: int
    total_repositories_indexed: int
    chunks_by_language: Dict[str, int] = field(default_factory=dict)
    average_chunk_size: int = 0
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass(frozen=True)
class SearchDiagnostics:
    """Timing and candidate statistics for one retrieval execution."""

    strategy_name: str
    retrieval_time_seconds: float
    reranking_time_seconds: float
    total_query_time_seconds: float
    vector_candidates: int = 0
    bm25_candidates: int = 0
    bm25_available: bool = False
    bm25_document_count: int = 0
    bm25_rebuilt: bool = False
    bm25_rebuild_source: str = ""
    bm25_rebuild_time_seconds: float = 0.0
    degradation_reason: str = ""
    merged_candidates: int = 0
    returned_results: int = 0


@dataclass(frozen=True)
class SearchExecution:
    """Structured search output for evaluation and dashboards."""

    results: List[SearchResult]
    diagnostics: SearchDiagnostics


class VectorService:
    """Coordinate embedding generation, persistence, and retrieval."""

    _bm25_init_lock = RLock()

    def __init__(
        self,
        repository_name: str,
        collection_name: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        chroma_manager: Optional[ChromaManager] = None,
    ):
        self.repository_name = repository_name
        self.logger = get_logger(self.__class__.__name__)
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.chroma_manager = chroma_manager or ChromaManager()
        self.collection_router = CollectionRouter(repository_name)
        self.query_classifier = QueryClassifier()
        self.ranking_pipeline = RankingPipeline()
        self.bm25_retriever = BM25Retriever()
        self.hybrid_pipeline = HybridRetrievalPipeline()
        self.collection_manager = CollectionManager(
            repository_name=repository_name,
            chroma_manager=self.chroma_manager,
            collection_name=collection_name or self.collection_router.collection_name_for_category(FileCategory.SOURCE_CODE),
        )
        self.collection_managers: Dict[str, CollectionManager] = {
            route.collection_name: CollectionManager(
                repository_name=repository_name,
                chroma_manager=self.chroma_manager,
                collection_name=route.collection_name,
            )
            for route in self.collection_router.build_routes().values()
        }

    def index_chunks(self, chunks: Sequence[CodeChunk], reindex: bool = False) -> IndexResult:
        """Embed and store chunks in ChromaDB."""

        start_time = datetime.now()
        valid_chunks = [chunk for chunk in chunks if chunk and chunk.content and chunk.content.strip()]
        self.logger.info(
            "Indexing %s chunks for repository '%s' (reindex=%s)",
            len(valid_chunks),
            self.repository_name,
            reindex,
        )
        embedded_chunks = self.embedding_manager.embed_chunks(valid_chunks)
        self.bm25_retriever.invalidate_index(self.repository_name)
        self.bm25_retriever.rebuild_index(self.repository_name, valid_chunks)
        grouped_chunks = self._group_embedded_chunks_by_collection(embedded_chunks)
        storage_result = {"stored": 0, "skipped": 0, "duplicates": 0}
        for collection_name, collection_chunks in grouped_chunks.items():
            manager = self._get_collection_manager(collection_name)
            result = manager.store_embedded_chunks(collection_chunks, reindex=reindex)
            for key in storage_result:
                storage_result[key] += result.get(key, 0)

        collection_stats = self._aggregate_collection_statistics()
        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            "Indexed repository '%s': stored=%s duplicates=%s duration=%.2fs",
            self.repository_name,
            storage_result.get("stored", 0),
            storage_result.get("duplicates", 0),
            duration,
        )

        return IndexResult(
            repository_name=self.repository_name,
            total_chunks=len(valid_chunks),
            embedded_chunks=len(embedded_chunks),
            stored_chunks=storage_result.get("stored", 0),
            skipped_chunks=storage_result.get("skipped", 0),
            duplicate_chunks=storage_result.get("duplicates", 0),
            total_files_processed=collection_stats.total_files_processed,
            total_repositories_indexed=collection_stats.total_repositories_indexed,
            chunks_by_language=collection_stats.chunks_by_language,
            average_chunk_size=collection_stats.average_chunk_size,
            duration_seconds=duration,
        )

    def reindex_chunks(self, chunks: Sequence[CodeChunk]) -> IndexResult:
        """Force a full re-index of the active repository collection."""

        return self.index_chunks(chunks, reindex=True)

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert Chroma distance values into a bounded similarity score."""

        return 1.0 / (1.0 + max(0.0, float(distance)))

    def _get_collection_manager(self, collection_name: str) -> CollectionManager:
        if collection_name not in self.collection_managers:
            self.collection_managers[collection_name] = CollectionManager(
                repository_name=self.repository_name,
                chroma_manager=self.chroma_manager,
                collection_name=collection_name,
            )
        return self.collection_managers[collection_name]

    def _group_embedded_chunks_by_collection(
        self,
        embedded_chunks: Sequence[EmbeddedChunk],
    ) -> Dict[str, List[EmbeddedChunk]]:
        grouped: Dict[str, List[EmbeddedChunk]] = {}
        for embedded_chunk in embedded_chunks:
            category = embedded_chunk.metadata.get("file_category", FileCategory.SOURCE_CODE)
            collection_name = self.collection_router.collection_name_for_category(category)
            grouped.setdefault(collection_name, []).append(embedded_chunk)
        return grouped

    def _aggregate_collection_statistics(self) -> CollectionStatistics:
        aggregate = CollectionStatistics(collection_name="multi_collection")
        chunks_by_language: Dict[str, int] = {}
        total_chunk_size = 0
        weighted_documents = 0
        file_paths = set()

        for manager in self.collection_managers.values():
            stats = manager.get_statistics()
            if stats.total_documents == 0:
                continue
            aggregate.total_documents += stats.total_documents
            aggregate.total_embeddings += stats.total_embeddings
            aggregate.total_chunks_created += stats.total_chunks_created
            aggregate.total_embeddings_stored += stats.total_embeddings_stored
            aggregate.total_repositories_indexed = max(
                aggregate.total_repositories_indexed,
                stats.total_repositories_indexed,
            )
            total_chunk_size += stats.average_chunk_size * stats.total_documents
            weighted_documents += stats.total_documents

            collection = manager.get_collection()
            data = collection.get(include=["metadatas"])
            for metadata in data.get("metadatas", []) or []:
                if isinstance(metadata, dict) and metadata.get("file_path"):
                    file_paths.add(metadata["file_path"])

            for language, count in stats.chunks_by_language.items():
                chunks_by_language[language] = chunks_by_language.get(language, 0) + count

        aggregate.total_files_processed = len(file_paths)
        aggregate.chunks_by_language = chunks_by_language
        aggregate.average_chunk_size = total_chunk_size // weighted_documents if weighted_documents else 0
        return aggregate

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Run semantic search over the indexed chunks."""

        return self.search_hybrid(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            enable_rerank=True,
        )

    def search_vector_only(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Run vector-only retrieval with the current ranking pipeline."""

        return self.search_vector_only_with_diagnostics(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        ).results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        enable_rerank: bool = True,
    ) -> List[SearchResult]:
        return self.search_hybrid_with_diagnostics(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            enable_rerank=enable_rerank,
        ).results

    def search_vector_only_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> SearchExecution:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        total_started = perf_counter()
        classification = self.query_classifier.classify(query)
        collection_names = self.collection_router.collections_for_intent(classification.intent)
        fetch_k = max(top_k, top_k * self.ranking_pipeline.config.candidate_multiplier)
        self.logger.info(
            "Running vector-only search for repository '%s': %s (intent=%s collections=%s)",
            self.repository_name,
            query,
            classification.intent,
            collection_names,
        )

        retrieval_started = perf_counter()
        search_results = self._vector_search_candidates(query, classification.intent, fetch_k)
        search_results = self.ranking_pipeline.rerank(
            query=query,
            classification=classification,
            candidates=search_results,
            top_k=top_k,
            explicit_filters=metadata_filters,
        )
        retrieval_time = perf_counter() - retrieval_started
        total_time = perf_counter() - total_started

        self.logger.info(
            "Vector-only search returned %s results for repository '%s'",
            len(search_results),
            self.repository_name,
        )
        isolated_results = self._filter_results_for_repository(search_results)
        return SearchExecution(
            results=isolated_results,
            diagnostics=SearchDiagnostics(
                strategy_name="vector",
                retrieval_time_seconds=retrieval_time,
                reranking_time_seconds=0.0,
                total_query_time_seconds=total_time,
                vector_candidates=len(search_results),
                returned_results=len(isolated_results),
            ),
        )

    def search_hybrid_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        enable_rerank: bool = True,
    ) -> SearchExecution:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        total_started = perf_counter()
        classification = self.query_classifier.classify(query)
        fetch_k = max(
            top_k,
            top_k * self.ranking_pipeline.config.candidate_multiplier,
            self.hybrid_pipeline.config.rerank_candidate_limit,
        )

        vector_lookup_started = perf_counter()
        vector_candidates = self._vector_search_candidates(query, classification.intent, fetch_k)
        bm25_state = self._ensure_bm25_index()
        bm25_candidates = self.bm25_retriever.search(self.repository_name, query, top_k=fetch_k)
        retrieval_lookup_time = perf_counter() - vector_lookup_started

        pipeline_result = self.hybrid_pipeline.run_with_diagnostics(
            query=query,
            classification=classification,
            vector_candidates=vector_candidates,
            bm25_candidates=bm25_candidates,
            top_k=top_k,
            explicit_filters=metadata_filters,
            enable_rerank=enable_rerank,
        )
        total_time = perf_counter() - total_started
        isolated_results = self._filter_results_for_repository(pipeline_result.results)
        diagnostics = SearchDiagnostics(
            strategy_name="hybrid_reranked" if enable_rerank else "hybrid",
            retrieval_time_seconds=retrieval_lookup_time + pipeline_result.retrieval_time_seconds,
            reranking_time_seconds=pipeline_result.reranking_time_seconds,
            total_query_time_seconds=total_time,
            vector_candidates=len(vector_candidates),
            bm25_candidates=len(bm25_candidates),
            bm25_available=bm25_state["available"],
            bm25_document_count=bm25_state["document_count"],
            bm25_rebuilt=bm25_state["rebuilt"],
            bm25_rebuild_source=bm25_state["source"],
            bm25_rebuild_time_seconds=bm25_state["rebuild_time_seconds"],
            degradation_reason=bm25_state["degradation_reason"],
            merged_candidates=pipeline_result.merged_candidates,
            returned_results=len(isolated_results),
        )

        self.logger.info(
            "Hybrid search returned %s results for repository '%s' (rerank=%s)",
            len(isolated_results),
            self.repository_name,
            enable_rerank,
        )
        return SearchExecution(results=isolated_results, diagnostics=diagnostics)

    def _ensure_bm25_index(self) -> Dict[str, Any]:
        """Ensure the repository BM25 index exists, lazily rebuilding from Chroma."""

        current_count = self.bm25_retriever.document_count(self.repository_name)
        if current_count > 0:
            return {
                "available": True,
                "document_count": current_count,
                "rebuilt": False,
                "source": "memory",
                "rebuild_time_seconds": 0.0,
                "degradation_reason": "",
            }

        with self._bm25_init_lock:
            current_count = self.bm25_retriever.document_count(self.repository_name)
            if current_count > 0:
                return {
                    "available": True,
                    "document_count": current_count,
                    "rebuilt": False,
                    "source": "memory",
                    "rebuild_time_seconds": 0.0,
                    "degradation_reason": "",
                }

            started = perf_counter()
            try:
                chunks = self._load_chunks_from_chroma()
                if not chunks:
                    reason = "BM25 index missing and no indexed Chroma documents were available"
                    self.logger.warning("%s for repository '%s'", reason, self.repository_name)
                    return {
                        "available": False,
                        "document_count": 0,
                        "rebuilt": False,
                        "source": "chroma",
                        "rebuild_time_seconds": perf_counter() - started,
                        "degradation_reason": reason,
                    }

                self.bm25_retriever.rebuild_index(self.repository_name, chunks)
                document_count = self.bm25_retriever.document_count(self.repository_name)
                reason = "" if document_count > 0 else "BM25 rebuild produced zero documents"
                self.logger.info(
                    "Lazily rebuilt BM25 index for repository '%s' from Chroma with %s documents",
                    self.repository_name,
                    document_count,
                )
                return {
                    "available": document_count > 0,
                    "document_count": document_count,
                    "rebuilt": True,
                    "source": "chroma",
                    "rebuild_time_seconds": perf_counter() - started,
                    "degradation_reason": reason,
                }
            except Exception as exc:
                reason = f"BM25 lazy rebuild failed: {exc}"
                self.logger.warning(reason, exc_info=True)
                return {
                    "available": False,
                    "document_count": 0,
                    "rebuilt": False,
                    "source": "chroma",
                    "rebuild_time_seconds": perf_counter() - started,
                    "degradation_reason": reason,
                }

    def _load_chunks_from_chroma(self) -> List[CodeChunk]:
        """Rehydrate CodeChunk objects from persisted Chroma documents and metadata."""

        chunks: List[CodeChunk] = []
        seen_ids = set()
        for manager in self.collection_managers.values():
            collection = manager.get_collection()
            if collection.count() == 0:
                continue
            data = collection.get(include=["documents", "metadatas"])
            ids = data.get("ids", []) or []
            documents = data.get("documents", []) or []
            metadatas = data.get("metadatas", []) or []
            for chunk_id, document, metadata in zip(ids, documents, metadatas):
                if not document or not str(document).strip():
                    continue
                metadata = metadata or {}
                stable_chunk_id = metadata.get("chunk_id") or chunk_id
                if stable_chunk_id in seen_ids:
                    continue
                metadata_repository = metadata.get("repository_name") or self.repository_name
                if metadata_repository != self.repository_name:
                    continue
                file_path = metadata.get("file_path") or ""
                if not file_path:
                    continue
                chunks.append(
                    CodeChunk(
                        chunk_id=stable_chunk_id,
                        file_name=metadata.get("file_name") or file_path.replace("\\", "/").split("/")[-1],
                        file_path=file_path,
                        language=metadata.get("language") or "unknown",
                        repository_name=self.repository_name,
                        content=str(document),
                        chunk_type=metadata.get("chunk_type") or "chunk",
                        class_name=metadata.get("class_name") or None,
                        function_name=metadata.get("function_name") or None,
                        start_line=int(metadata.get("start_line", 0) or 0),
                        end_line=int(metadata.get("end_line", 0) or 0),
                        metadata=dict(metadata),
                    )
                )
                seen_ids.add(stable_chunk_id)
        return chunks

    def _vector_search_candidates(
        self,
        query: str,
        intent: str,
        fetch_k: int,
    ) -> List[SearchResult]:
        collection_names = self.collection_router.collections_for_intent(intent)
        embedding = self.embedding_manager.embedding_service.embed_text(query)
        search_results: List[SearchResult] = []
        for collection_name in collection_names:
            manager = self._get_collection_manager(collection_name)
            if manager.get_collection().count() == 0:
                continue
            results = manager.query(
                embedding,
                top_k=fetch_k,
                where={"repository_name": self.repository_name},
            )
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            ids = (results.get("ids") or [[]])[0]
            for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
                metadata = metadata or {}
                metadata_repository = metadata.get("repository_name") or self.repository_name
                if metadata_repository != self.repository_name:
                    self.logger.warning(
                        "Discarding vector candidate from repository '%s' while searching repository '%s' (chunk_id=%s)",
                        metadata_repository,
                        self.repository_name,
                        chunk_id,
                    )
                    continue
                vector_score = self._distance_to_similarity(distance)
                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        similarity_score=vector_score,
                        distance=float(distance),
                        file_name=metadata.get("file_name", ""),
                        file_path=metadata.get("file_path", ""),
                        language=metadata.get("language", ""),
                        class_name=metadata.get("class_name", ""),
                        function_name=metadata.get("function_name", ""),
                        start_line=int(metadata.get("start_line", 0) or 0),
                        end_line=int(metadata.get("end_line", 0) or 0),
                        document=document or "",
                        collection_name=collection_name,
                        repository_name=metadata_repository,
                        original_vector_score=vector_score,
                        fusion_score=vector_score,
                        adjusted_score=vector_score,
                        query_intent=intent,
                        metadata=metadata,
                    )
                )
        return search_results

    def _filter_results_for_repository(self, results: Sequence[SearchResult]) -> List[SearchResult]:
        isolated: List[SearchResult] = []
        for result in results:
            metadata = getattr(result, "metadata", {}) or {}
            result_repository = getattr(result, "repository_name", "") or metadata.get("repository_name") or self.repository_name
            if result_repository != self.repository_name:
                self.logger.warning(
                    "Discarding retrieved result from repository '%s' while searching repository '%s' (chunk_id=%s)",
                    result_repository,
                    self.repository_name,
                    getattr(result, "chunk_id", ""),
                )
                continue
            if not getattr(result, "repository_name", ""):
                result.repository_name = self.repository_name
            isolated.append(result)
        return isolated

    def get_statistics(self) -> Dict[str, Any]:
        """Return repository-level indexing statistics."""

        stats: CollectionStatistics = self._aggregate_collection_statistics()
        return {
            "repository_name": self.repository_name,
            "total_repositories_indexed": stats.total_repositories_indexed,
            "total_files_processed": stats.total_files_processed,
            "total_chunks_created": stats.total_chunks_created,
            "total_embeddings_stored": stats.total_embeddings_stored,
            "chunks_by_language": stats.chunks_by_language,
            "average_chunk_size": stats.average_chunk_size,
            "collection_name": stats.collection_name,
            "collections": list(self.collection_managers.keys()),
        }
