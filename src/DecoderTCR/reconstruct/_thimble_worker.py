"""Run thimble with process-local paths to bundled reference data."""

from __future__ import annotations

from ._resources import data_path


def main() -> None:
    """Configure bundled Stitchr data paths and run thimble."""
    from Stitchr import stitchrfunctions

    root = data_path("stitchr")
    stitchrfunctions.data_dir = str(root)
    stitchrfunctions.additional_genes_file = str(root / "additional-genes.fasta")
    stitchrfunctions.linkers_file = str(root / "linkers.tsv")
    stitchrfunctions.gui_examples_dir = str(root / "GUI-Examples")

    from Stitchr import thimble

    thimble.main()


if __name__ == "__main__":
    main()
