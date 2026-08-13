#!/usr/bin/env python3
"""Convert multi-page ODG to per-page SVGs with clean fonts.

Usage:
    python3 odg2svg.py ob_figures.odg
    python3 odg2svg.py ob_figures.odg --prefix fig --output ./out/

Output: fig1.svg, fig2.svg, ...
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def match_close(svg: str, start: int) -> int:
    """Find matching </g> for <g> at position start (depth starts at 1)."""
    depth = 1
    i = start
    while i < len(svg):
        no = svg.find("<g", i)
        nc = svg.find("</g>", i)
        if no == -1:
            no = 10**9
        if nc == -1:
            nc = 10**9
        if min(no, nc) == 10**9:
            break
        if no < nc:
            te = svg.find(">", no)
            tag = svg[no : te + 1] if te != -1 else ""
            if not tag.rstrip().endswith("/>"):
                depth += 1
            i = te + 1 if te != -1 else no + 2
        else:
            depth -= 1
            if depth == 0:
                return nc + 4
            i = nc + 4
    return -1


def convert(odg_path: Path, prefix: str = "fig", out_dir: Path | None = None) -> list[Path]:
    if out_dir is None:
        out_dir = Path.cwd()

    print(f"Converting {odg_path} ...", file=sys.stderr)
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "svg", str(odg_path.resolve())],
        cwd=out_dir,
        check=True,
        capture_output=True,
    )

    svg_path = out_dir / odg_path.with_suffix(".svg").name
    if not svg_path.exists():
        print(f"ERROR: {svg_path} not produced", file=sys.stderr)
        return []

    svg = svg_path.read_text()

    svg = re.sub(
        r"<font id=\"EmbeddedFont_\d+\"[^>]*>.*?</font>", "", svg, flags=re.DOTALL
    )
    svg = svg.replace(
        'font-family="Nimbus Roman embedded"', 'font-family="Nimbus Roman, serif"'
    )
    svg = svg.replace(
        'font-family="Nimbus Roman"', 'font-family="Nimbus Roman, serif"'
    )

    sg = svg.find('<g class="SlideGroup">')
    containers = [
        m.start()
        for m in re.finditer(r'<g id="container-id\d+">', svg)
    ]
    if not containers:
        print("ERROR: no container-id elements found", file=sys.stderr)
        svg_path.unlink()
        return []

    after_sg = svg.find("<g", sg + len('<g class="SlideGroup">') + 1)
    header = svg[:after_sg]

    sg_close = match_close(svg, sg)
    svg_end = svg.rfind("</svg>")
    footer = svg[sg_close:svg_end] + "\n</svg>"

    outputs = []
    for i, start in enumerate(containers, 1):
        end = match_close(svg, start)
        if end == -1:
            print(f"WARNING: container {i} has no matching </g>", file=sys.stderr)
            continue
        page = header + "\n" + svg[start:end] + footer
        out_path = out_dir / f"{prefix}{i}.svg"
        out_path.write_text(page)
        outputs.append(out_path)

    svg_path.unlink()

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Convert ODG to per-page SVGs")
    parser.add_argument("odg", type=Path, help="Input ODG file")
    parser.add_argument("--prefix", default="fig", help="Output filename prefix (default: fig)")
    parser.add_argument("--output", "-o", type=Path, help="Output directory (default: same as input)")
    args = parser.parse_args()

    if not args.odg.exists():
        print(f"ERROR: {args.odg} not found", file=sys.stderr)
        sys.exit(1)

    outputs = convert(args.odg, args.prefix, args.output)
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
