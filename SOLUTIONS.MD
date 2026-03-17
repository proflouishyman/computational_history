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

## 2026-03-17 - Reorder post folders with numeric prefixes

### Problem
Post folders were not explicitly numbered, which made intended publication order unclear. The desired order is: first CHEAP CODE, second Historians + stats.

### Solution
Renamed post directories to encode order in folder names:

- `posts/cheap-code/` -> `posts/01-cheap-code/`
- `posts/historians-stats/` -> `posts/02-historians-stats/`

Updated `README.md` current post links to the new numbered paths so repository navigation and linking follow the intended sequence.

## 2026-03-17 - Add Post 4 code package for sharing

### Problem
The fourth post ("AI as Intern") had working code in a separate local project path, but it was not present in this publication repository under the standard post folder contract. That made the code harder to share consistently.

### Solution
Added a numbered post folder and imported the script into the repository contract:

- Created `posts/04-ai-as-intern/` with `source-docx/`, `code/`, `method/`, and `data/`
- Copied `shallow_reading_local.py` into `posts/04-ai-as-intern/code/`
- Added `posts/04-ai-as-intern/README.md` for navigation and link placeholders
- Added `posts/04-ai-as-intern/method/SHALLOW_READING_WORKFLOW.md` documenting input/output and schema
- Updated root `README.md` to include `posts/04-ai-as-intern/` in current posts
