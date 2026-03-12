# Text-Only Version of LaTeX Paper

This folder contains a text-only version of the thesis paper with all images and figures removed.

## Key Differences from Original

1. **No Images**: All `\includegraphics` commands removed
2. **No Figure Environments**: All `\begin{figure}...\end{figure}` blocks removed
3. **No List of Figures**: Removed from table of contents
4. **Simplified Packages**: Removed graphicx and image-related packages

## File Structure

- `main.tex` - Main document (simplified, no image packages)
- `chapter*.tex` - All chapter files with figures removed
- Other `.tex` files - Copied from original (Declaration, Acknowledgement, etc.)
- `ref.bib` - Bibliography file (same as original)

## Compilation

To compile this version:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Or use the quick compile:

```bash
pdflatex -interaction=nonstopmode main.tex && biber main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

## Output

- **Pages**: ~70 pages (compared to 76 in original with figures)
- **Size**: ~350KB (compared to ~7.5MB with images)
- **Content**: All text, tables, equations, and bibliography preserved

## Use Cases

This version is useful for:
- Quick text review without loading images
- Faster compilation times
- Smaller file size for email/sharing
- Text-only proofreading
- Accessibility (screen readers)
