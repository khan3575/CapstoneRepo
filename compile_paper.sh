#!/bin/bash
# Compile IEEE paper to PDF

echo "Compiling IEEE format paper..."

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found. Install with:"
    echo "  sudo apt-get install texlive-full"
    exit 1
fi

cd "$(dirname "$0")"

# Compile (run twice for references)
pdflatex paper_ieee_format.tex
bibtex paper_ieee_format
pdflatex paper_ieee_format.tex
pdflatex paper_ieee_format.tex

# Clean up auxiliary files
rm -f *.aux *.log *.out *.bbl *.blg

echo "✅ Paper compiled successfully!"
echo "Output: paper_ieee_format.pdf"
