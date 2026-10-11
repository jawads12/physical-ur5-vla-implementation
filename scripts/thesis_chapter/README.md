# Thesis chapter package

Start with `experimental_methodology.docx` for editing or `experimental_methodology.pdf` for reading. The chapter contains approximately 7,000 words, three system diagrams, and a plot of saved checkpoint comparisons.

The focus is how the UR5 platform reached its current experimental stage. Section 15 supplies a reusable record for future tasks and experiments. Documented results, implemented mechanisms, and future evaluation are distinguished throughout.

Files:

- `experimental_methodology.docx`: editable Word document with figures and tables.
- `experimental_methodology.pdf`: formatted reading copy.
- `experimental_methodology.md`: editable source text.
- `experimental_methodology.html`: browser and print version; keep the figures folder alongside it.
- `figures/`: all four figures as PNG, SVG, and PDF; system diagrams also include editable Graphviz source.
- `checkpoint_comparison.csv`: metrics from all 16 saved checkpoint reports.
- `evidence_register.json`: 54 local source paths with SHA-256 hashes, grouped by chapter reference.
- `build_chapter.py`: rebuilds figures, data table, source register, HTML, and Word document from local source records. It does not execute robot-control code.

To extend the thesis, edit the Markdown source, rebuild with the documented Python dependencies, and export the Word document to PDF. The source paths in the evidence register refer to the inspected workstation and will need adaptation on another computer.

Before thesis submission, apply the university formatting requirements, add scholarly references, and add physical experiment outcomes from actual trial records. The source datasets and model weights are not bundled in this package.
