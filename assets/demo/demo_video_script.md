# RepoMind AI Demo Video Script

Target duration: 2-3 minutes

## Structure

| Segment | Time | Purpose |
| --- | ---: | --- |
| Introduction | 0:00-0:20 | Introduce RepoMind AI and the value proposition. |
| Problem | 0:20-0:45 | Explain why repository onboarding is difficult. |
| Features | 0:45-1:15 | Highlight the main product capabilities. |
| Demo | 1:15-2:15 | Walk through the core UI flow. |
| Architecture | 2:15-2:40 | Briefly explain the RAG system behind the product. |
| Closing | 2:40-3:00 | Summarize impact and project readiness. |

## Narration Script

Hi, this is RepoMind AI, an AI-powered repository understanding assistant built for developers who need to understand unfamiliar codebases quickly.

When someone joins a new project, they usually spend hours searching through files, reading code out of order, and trying to guess how different modules connect. Standard keyword search can find text, but it does not explain architecture, onboarding paths, or implementation details with evidence.

RepoMind AI solves that by turning a GitHub repository into an interactive knowledge system. It clones and indexes a repository, parses the source code, creates chunks, stores embeddings, combines semantic search with BM25 keyword retrieval, and generates grounded answers using an LLM.

Here is the main Repository Intelligence dashboard. At the top, RepoMind shows repository-level statistics such as files, chunks, embeddings, primary language, repository size, and the last indexed timestamp. The retrieval mode can be switched between Fast Mode for quick interactive questions and Deep Analysis for higher-quality answers with reranking.

The primary workflow is Ask Repository. I can ask a question like, "Explain the repository architecture." RepoMind answers using retrieved code context from the selected repository only. The response is grounded, and it includes evidence such as referenced files, functions, classes, citations, and retrieval diagnostics.

RepoMind also generates structured reports. For example, an architecture report summarizes how the system is organized, provides more detailed explanation, and keeps the evidence visible so the answer is not just a black box.

For deeper exploration, the Module Explorer lets me select a file, folder, class, or module and generate a focused explanation. Repository Health gives quick metadata diagnostics, including language distribution, files, configuration, documentation, and deeper analysis signals. The Retrieval Evaluation page supports measuring retrieval quality with benchmark controls.

Behind the scenes, RepoMind uses a retrieval-augmented generation pipeline. A GitHub repository is cloned, scanned, parsed, chunked, embedded with SentenceTransformers, stored in ChromaDB, matched with BM25, fused through hybrid retrieval, optionally reranked with a CrossEncoder, and then passed through context and prompt builders before the LLM generates a grounded response.

The result is a polished developer tool for repository onboarding, architecture understanding, and evidence-backed codebase Q&A. It is designed to be practical for real engineering workflows and clear enough to showcase as a portfolio AI project.

## Demo Flow Checklist For Recording

1. Open RepoMind AI.
2. Show Repository Dashboard.
3. Pause briefly on Repository Overview cards.
4. Scroll to Ask Repository.
5. Ask: `Explain the repository architecture`.
6. Show Fast Mode selected.
7. Switch to Deep Analysis.
8. Show Generated Report.
9. Show Module Explorer.
10. Show Repository Health.
11. Show Retrieval Evaluation.
12. Show Export Report buttons.
13. Close with value proposition.
