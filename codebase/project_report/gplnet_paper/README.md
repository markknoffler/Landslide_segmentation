# GeoPhysicsLandslideNet — Paper & Project Report

Authors: **Samreedh Bhuyan**, guide **Dr. Anil Earnest** (CSIR-4PI)  
Institution: KIIT (School of Computer Engineering)

## Contents

| File | Description |
|------|-------------|
| `paper.tex` / `paper.pdf` | Two-column research paper (conference style) |
| `report.tex` / `report.pdf` | Project report with official 4PI letterheads + acknowledgements |
| `letterhead/` | Filled thesis certificate & cover page (DOCX/PDF) |
| `assets/` | Metrics JSON, generated figures, neist-extracted panels |
| `figures/` | Figures used by LaTeX |
| `bib/refs.bib` | Bibliography |

## Final model results (source)

- Bijie: `from_first_steps/outputs_absolute_final_fully_novel_complete` (symlink `outputs_final_bijie`)
- Landslide4Sense: `from_first_steps/outputs_step3_l4s` (symlink `outputs_final_l4s`)
- Legacy runs archived under `from_first_steps/prev_legacy_results/`
- Architecture doc: `from_first_steps/model_architecture.md`

## Rebuild

```bash
cd gplnet_paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
pdflatex report.tex && pdflatex report.tex
```
