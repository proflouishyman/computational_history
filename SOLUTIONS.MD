# Solutions Log

## 2026-03-07 - Initialize Computational History repo structure and post organization

### Problem
The project root did not have a reusable repository structure for publication artifacts, and the first two post drafts were stored outside this working root, making it hard to link code/method/data consistently from Substack.

### Solution
Initialized a Git repository and created a standard folder contract for Computational History work:

- Added top-level directories: `posts/`, `methods/`, `prompts/`, `data-dictionaries/`, `assets/images/`
- Copied post drafts into post-specific folders:
  - `posts/cheap-code/source-docx/CHEAP CODE.docx`
  - `posts/historians-stats/source-docx/hyman-blurb-historians-stats.docx`
- Added per-post scaffolding directories: `code/`, `method/`, `data/`
- Added documentation:
  - `README.md` (repository contract + linking pattern)
  - post-level READMEs for both posts
  - `.gitignore` for common local artifacts

This establishes a consistent, repeatable contract for linking `Code`, `Method + prompts`, and `Data availability` in published posts.

## 2026-03-07 - Add CHEAP CODE pipeline files into post contract

### Problem
The CHEAP CODE post had references to implementation files outside this repository, which made publication links and reproducibility pointers inconsistent.

### Solution
Copied the implementation files into the CHEAP CODE post folder while preserving the repository contract:

- Added `posts/cheap-code/code/probate_audit_pipeline.py`
- Added `posts/cheap-code/method/AUDIT_LOGIC.md`
- Updated `posts/cheap-code/README.md` to point to concrete code/method files

This keeps all post-linked reproducibility artifacts inside the same publication repository.
