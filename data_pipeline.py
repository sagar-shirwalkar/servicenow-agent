#!/usr/bin/env python3
"""
Optimized ServiceNowDocs parser with rolling-window chunking and fixed path categorization.
"""

from pathlib import Path
from markdown_it import MarkdownIt

md = MarkdownIt()

# Map directory names to high-level categories
TOPIC_CATEGORIES = {
    "api implementation and reference": "api",
    "workflow data fabric": "api",
    "app development and low-code": "code",
    "build workflows": "code",
    "building applications": "code",
    "mobile platform": "code",
    "administer the servicenow ai platform": "docs",
    "configure user experiences": "docs",
    "enable ai experiences": "docs",
    "extend servicenow ai platform capabilities": "docs",
    "platform analytics": "docs",
    "secure your instance": "docs",
    "account lifecycle events": "docs",
    "customer service management": "docs",
    "employee service management": "docs",
    "field service management": "docs",
    "it service management": "docs",
    "it operations management": "docs",
    "security operations": "docs",
    "governance, risk, and compliance": "docs",
    "it asset management": "docs",
    "strategic portfolio management": "docs",
    "enterprise architecture": "docs",
    "cloud observability": "docs",
    "conversational interfaces": "docs",
    "cloud governance suite": "docs",
    "core business suite": "docs",
    "environmental, social, and esg management": "docs",
    "financial services operations": "docs",
    "healthcare and life sciences": "docs",
    "industrial connected workforce": "docs",
    "crm and industry products": "docs",
    "impact": "docs",
    "manufacturing commercial operations": "docs",
    "operational technology": "docs",
    "product directory": "docs",
    "product support for technology": "docs",
    "finance and supply chain": "docs",
    "retail": "docs",
    "service exchange": "docs",
    "service management": "docs",
    "technology industry": "docs",
    "telecommunications, media, and technology": "docs",
    "telecommunications network inventory": "docs",
    "telecommunications service operations management": "docs",
    "sales and order management": "docs",
    "public sector digital services": "docs",
}

def _categorize_by_path(file_path: Path, repo_root: Path) -> tuple[str, str]:
    """Fixed: Checks parent directories correctly, not just the leaf folder."""
    try:
        relative = file_path.relative_to(repo_root)
        parts = [p.lower() for p in relative.parts[:-1]]
        
        for part in parts:
            normalized = part.replace("_", " ").strip()
            if normalized in TOPIC_CATEGORIES:
                return TOPIC_CATEGORIES[normalized], normalized
                
        topic = parts[0] if parts else "general"
        return "docs", topic.replace("_", " ").strip()
    except ValueError:
        return "docs", "unknown"

def _extract_semantic_chunks(text: str, file_path: Path, repo_root: Path) -> list[dict]:
    """Rolling window chunking: groups text logically and guarantees size limits."""
    category, topic = _categorize_by_path(file_path, repo_root)
    tokens = md.parse(text)
    
    chunks = []
    text_buffer = ""
    current_heading = "Overview"
    
    MAX_CHUNK_CHARS = 2000  # ~500 tokens. Very safe for nomic-embed-text
    OVERLAP_CHARS = 200    # Context overlap for semantic continuity
    
    def _flush_buffer():
        nonlocal text_buffer
        if not text_buffer.strip():
            return
            
        # Split buffer if it exceeds limits, preferring newline or sentence boundaries
        while len(text_buffer) > MAX_CHUNK_CHARS:
            split_idx = text_buffer.rfind('\n', 0, MAX_CHUNK_CHARS)
            if split_idx < MAX_CHUNK_CHARS // 2:
                split_idx = text_buffer.rfind('. ', 0, MAX_CHUNK_CHARS)
            if split_idx < MAX_CHUNK_CHARS // 2:
                split_idx = MAX_CHUNK_CHARS
                
            chunk_text = text_buffer[:split_idx].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text, "category": category, "topic": topic,
                    "source_file": str(file_path.relative_to(repo_root)),
                    "heading": current_heading, "is_code": False
                })
            # Keep overlap for context
            text_buffer = text_buffer[split_idx - OVERLAP_CHARS:].strip() if split_idx > OVERLAP_CHARS else text_buffer[split_idx:].strip()
            
        if text_buffer.strip():
            chunks.append({
                "text": text_buffer.strip(), "category": category, "topic": topic,
                "source_file": str(file_path.relative_to(repo_root)),
                "heading": current_heading, "is_code": False
            })
            text_buffer = ""

    for token in tokens:
        if token.type == 'heading_open':
            _flush_buffer()
            if token.children and token.children[0].type == 'inline':
                current_heading = token.children[0].content.strip()
                
        elif token.type == 'fence':
            _flush_buffer()
            lang = token.info or "text"
            code_content = f"```{lang}\n{token.content}\n```"
            is_code = lang in ["javascript", "python", "json", "yaml", "bash", "shell"]
            cat = "code" if is_code else category
            
            # Code blocks get their own isolated chunk
            chunks.append({
                "text": code_content, "category": cat, "topic": topic,
                "source_file": str(file_path.relative_to(repo_root)),
                "heading": current_heading, "is_code": is_code
            })
            
        elif token.type == 'inline' and token.content.strip():
            text_buffer += token.content.strip() + "\n"
            
        elif token.type == 'paragraph_close':
            text_buffer += "\n"

    _flush_buffer()
    return chunks

def parse_and_categorize(repo_path: str) -> dict[str, list[dict]]:
    repo_root = Path(repo_path)
    if not repo_root.exists():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")
    
    dataset = {"api": [], "code": [], "docs": []}
    md_files = list(repo_root.rglob("*.md"))
    print(f"Found {len(md_files)} Markdown files in {repo_path}")
    
    for i, md_file in enumerate(md_files):
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1}/{len(md_files)} files...")
        try:
            text = md_file.read_text(encoding="utf-8")
            chunks = _extract_semantic_chunks(text, md_file, repo_root)
            for chunk in chunks:
                dataset[chunk["category"]].append(chunk)
        except Exception as e:
            continue
            
    for cat, chunks in dataset.items():
        print(f"   {cat}: {len(chunks)} chunks")
    return dataset
