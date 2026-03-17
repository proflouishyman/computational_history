# computational-history

Repository for code, methods, prompts, and reproducibility artifacts behind the Computational History publication.

## Structure

- `posts/`: one folder per post project
- `methods/`: reusable workflows and notes
- `prompts/`: prompt text and extraction specs
- `data-dictionaries/`: schemas and variable definitions
- `assets/images/`: publication visuals used across posts

## Post folders

Each post folder should use the same contract:

- `source-docx/`: source writing draft
- `code/`: scripts/notebooks used for this post
- `method/`: methodology and reproducibility notes
- `data/`: derived data (not restricted raw scans)
- `README.md`: quick navigation for that post

## Current posts

- `posts/01-cheap-code/` (Post 1)
- `posts/02-historians-stats/` (Post 2)
- `posts/04-ai-as-intern/` (Post 4)

## Suggested linking pattern in posts

- `Code:` direct link to `posts/<slug>/code/`
- `Method + prompts:` direct link to `posts/<slug>/method/`
- `Data availability:` direct link to `posts/<slug>/data/`
