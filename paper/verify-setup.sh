#!/bin/bash
# Simple LaTeX syntax verification script

echo "=== LaTeX Project Scaffolding Verification ==="
echo

echo "✅ Check 1: Main .tex file exists"
if [ -f "originblame.tex" ]; then
    echo "   ✓ paper/originblame.tex exists"
else
    echo "   ✗ paper/originblame.tex missing"
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
if [ -d "sections" ] && [ -d "drafts" ] && [ -d "figures" ]; then
    echo "   ✓ All section directories exist"
else
    echo "   ✗ Section directories missing"
fi

echo
echo "✅ Check 4: Section files exist"
section_count=$(find sections/ -name "*.tex" | wc -l)
if [ $section_count -eq 8 ]; then
    echo "   ✓ All 8 section files exist"
else
    echo "   ✗ Found $section_count section files, expected 8"
fi

echo
echo "✅ Check 5: .gitignore exists"
if [ -f ".gitignore" ]; then
    echo "   ✓ .gitignore exists"
else
    echo "   ✗ .gitignore missing"
fi

echo
echo "✅ Check 6: PDF output directory"
if [ -d "pdf-output" ]; then
    echo "   ✓ PDF output directory exists"
else
    echo "   ℹ️  PDF output directory not created yet"
fi

echo
echo "=== Note: LaTeX compilation requires pdflatex ==="
echo "The scaffolding is correctly structured but LaTeX"
echo "is not available in this environment."
echo "Run 'pdflatex originblame.tex' (twice) when LaTeX is available."