#!/bin/bash
# Build SwaraLM_paper.pdf from SwaraLM_paper.md + references.bib (pandoc >= 3.1, xelatex, TeX Gyre Termes).
set -e
cd "$(dirname "$0")"  # writes SwaraLM_paper.pdf next to the sources
pandoc SwaraLM_paper.md -o SwaraLM_paper.tex --standalone --columns=200 --template=tpl.latex --citeproc --bibliography=references.bib \
  -M link-citations=true -V documentclass=article -V papersize=a4 -V fontsize=11pt -V geometry:margin=1in \
  -V mainfont="TeX Gyre Termes" -V monofont="DejaVu Sans Mono" -V monofontoptions="Scale=0.82" -V linestretch=1.08 \
  -V colorlinks=true -V linkcolor=black -V citecolor="[rgb]{0.10,0.20,0.50}" -V urlcolor="[rgb]{0.10,0.20,0.50}" \
  -H header.tex
xelatex -interaction=nonstopmode -halt-on-error SwaraLM_paper.tex > build.log 2>&1
xelatex -interaction=nonstopmode -halt-on-error SwaraLM_paper.tex >> build.log 2>&1
