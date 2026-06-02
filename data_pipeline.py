import os
from markdown_it import MarkdownIt
from pathlib import Path

#
# Standard text splitters destroy code blocks and API schemas. 
# We must parse the Markdown Abstract Syntax Tree (AST) to keep logical blocks intact.
# 

#
# Tradeoff: AST parsing is slower than regex splitting, but it guarantees that a 50-line 
# JavaScript GlideRecord snippet or a complex JSON OpenAPI schema is kept as a single, 
# retrievable unit, preventing "semantic leakage."
# 

# Initialize Markdown parser
md = MarkdownIt()

def parse_and_categorize(repo_path: str):
    """Parses MD files and categorizes them into Docs, API, and Code."""
    dataset = {"docs": [], "api": [], "code": []}
    
    for md_file in Path(repo_path).rglob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        tokens = md.parse(text)
        
        current_chunk = []
        current_category = "docs"
        
        for token in tokens:
            # Detect headers to create semantic boundaries
            if token.type == 'heading_open':
                if current_chunk:
                    _save_chunk(current_chunk, current_category, dataset)
                    current_chunk = []
                
                # Categorize based on header text
                header_text = token.children[0].content.lower() if token.children else ""
                if "api" in header_text or "rest" in header_text or "openapi" in header_text:
                    current_category = "api"
                elif "code" in header_text or "script" in header_text or "example" in header_text:
                    current_category = "code"
                else:
                    current_category = "docs"
            
            # Extract raw text and code blocks
            if token.type == 'inline':
                current_chunk.append(token.content)
            elif token.type == 'fence': # Code block
                current_chunk.append(f"```{token.info}\n{token.content}\n```")
                
        if current_chunk:
            _save_chunk(current_chunk, current_category, dataset)
            
    return dataset

def _save_chunk(chunk, category, dataset):
    text = "\n".join(chunk).strip()
    if len(text) > 50: # Filter out noise
        dataset[category].append(text)

# Usage:
# data = parse_and_categorize("./ServiceNowDocs-australia")
