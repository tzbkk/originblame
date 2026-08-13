#!/bin/bash
# Simple LaTeX project verification script for the full paper.
# Run from paper/full/:  bash verify-setup.sh

echo "=== LaTeX Project Scaffolding Verification ==="
echo

echo "✅ Check 1: Main .tex file exists"
if [ -f "originblame.tex" ]; then
    echo "   ✓ paper/full/originblame.tex exists"
else
    echo "   ✗ paper/full/originblame.tex missing"
fi

echo
echo "✅ Check 2: Style files exist"
if [ -f "acl.sty" ] && [ -f "acl_natbib.bst" ]; then
    echo "   ✓ ACL style files present"
else
    echo "   ✗ ACL style files missing"
fi

echo
echo "✅ Check 3: Section directories exist"
if [ -d "sections" ] && [ -d "figures" ] && [ -d "tables" ]; then
    echo "   ✓ All directories exist (sections/, figures/, tables/)"
else
    echo "   ✗ Directory missing (expected sections/, figures/, tables/)"
fi

echo
echo "✅ Check 4: Section files exist"
section_count=$(find sections/ -name "*.tex" | wc -l)
if [ $section_count -eq 11 ]; then
    echo "   ✓ All 11 section files exist"
else
    echo "   ✗ Found $section_count section files, expected 11"
fi

echo
echo "✅ Check 5: .gitignore exists"
if [ -f ".gitignore" ]; then
    echo "   ✓ .gitignore exists"
else
    echo "   ✗ .gitignore missing"
fi

echo
echo "=== Note: LaTeX compilation requires pdflatex ==="
echo "Run from this directory:"
echo "  pdflatex originblame.tex && bibtex originblame && pdflatex originblame.tex && pdflatex originblame.tex"
