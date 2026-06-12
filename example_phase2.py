"""
Phase 2 Example: Code Parsing and Chunking

Demonstrates how to use the Phase 2 parsing system to extract chunks from a repository.

Example Usage:
    python example_phase2.py /path/to/repository "repository_name"
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from parser.parser_manager import ParserManager, ChunkAggregator


def main():
    """Main example function"""
    
    # Example: Parse the RAG application itself
    repo_path = str(project_root)
    repo_name = "RAG_application"
    
    print("="*70)
    print("PHASE 2: CODE PARSING AND INTELLIGENT CHUNKING")
    print("="*70)
    print()
    
    # Step 1: Initialize parser manager
    print(f"📁 Repository: {repo_name}")
    print(f"📍 Path: {repo_path}")
    print()
    
    manager = ParserManager(repo_path, repo_name)
    
    # Step 2: Parse repository
    print("🔍 Scanning and parsing repository...")
    chunks, report = manager.parse_repository()
    
    # Step 3: Display report
    print()
    manager.print_report()
    
    # Step 4: Analyze chunks by language
    print("📊 CHUNKS BY LANGUAGE:")
    print("-" * 70)
    
    for language in sorted(set(c.language for c in chunks)):
        lang_chunks = [c for c in chunks if c.language == language]
        print(f"\n{language.upper()}: {len(lang_chunks)} chunks")
        
        # Show sample chunks for this language
        for chunk in lang_chunks[:3]:
            display_name = chunk.get_display_name()
            file_name = Path(chunk.file_path).name
            lines = chunk.end_line - chunk.start_line + 1 if chunk.start_line else 0
            
            print(f"  • {display_name:<40} | {file_name:<20} | {lines} lines")
        
        if len(lang_chunks) > 3:
            print(f"  ... and {len(lang_chunks) - 3} more")
    
    # Step 5: Analyze chunks by type
    print()
    print("📋 CHUNKS BY TYPE:")
    print("-" * 70)
    
    type_counts = {}
    for chunk in chunks:
        type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1
    
    for chunk_type, count in sorted(type_counts.items()):
        print(f"  {chunk_type:<15}: {count:>4} chunks")
    
    # Step 6: Show example chunks
    print()
    print("📌 EXAMPLE CHUNKS:")
    print("-" * 70)
    
    if chunks:
        # Show first 5 chunks
        for i, chunk in enumerate(chunks[:5], 1):
            print(f"\nChunk {i}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Name: {chunk.get_display_name()}")
            print(f"  File: {chunk.file_path}")
            print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
            print(f"  Size: {chunk.get_size_info()['characters']} chars")
            print(f"  Type: {chunk.chunk_type}")
            if chunk.docstring:
                docstring_preview = chunk.docstring[:100] + "..." if len(chunk.docstring) > 100 else chunk.docstring
                print(f"  Doc: {docstring_preview}")
    
    # Step 7: Demonstrate search functionality
    print()
    print("🔎 SEARCH EXAMPLES:")
    print("-" * 70)
    
    search_terms = ["def", "class", "function"]
    
    for term in search_terms[:2]:  # Just show 2 examples
        results = manager.search_chunks(term)
        print(f"\nSearching for '{term}': Found {len(results)} chunks")
        if results:
            for chunk in results[:2]:
                print(f"  • {chunk.get_display_name()} in {Path(chunk.file_path).name}")
    
    # Step 8: Demonstrate class index
    print()
    print("🏗️  CLASS STRUCTURE:")
    print("-" * 70)
    
    aggregator = ChunkAggregator(chunks)
    class_index = aggregator.build_class_index()
    
    for class_name, class_chunks in sorted(class_index.items())[:5]:
        print(f"\nClass: {class_name}")
        methods = [c for c in class_chunks if c.chunk_type == "method"]
        print(f"  Methods: {len(methods)}")
        for method in methods[:3]:
            print(f"    • {method.function_name}()")
        if len(methods) > 3:
            print(f"    • ... and {len(methods) - 3} more")
    
    # Step 9: Save chunks to file
    print()
    print("💾 SAVING CHUNKS:")
    print("-" * 70)
    
    output_file = project_root / "data" / "chunks.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if manager.save_chunks(str(output_file)):
        print(f"✅ Chunks saved to: {output_file}")
        print(f"   File size: {output_file.stat().st_size / 1024:.2f} KB")
    else:
        print("❌ Failed to save chunks")
    
    print()
    print("="*70)
    print(f"✅ PARSING COMPLETE: {len(chunks)} chunks extracted")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
