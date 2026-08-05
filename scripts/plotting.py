"""Non-interactive figure output for experiment scripts."""

from pathlib import Path

import matplotlib.pyplot as plt


class FigureWriter:
    """Save open Matplotlib figures with stable, sequential names."""

    def __init__(self, output_dir: str | Path, prefix: str, *, dpi: int = 300, image_format: str = "png") -> None:
        self.output_dir = Path(output_dir) / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.dpi = dpi
        self.image_format = image_format
        self.counter = 0
        plt.switch_backend("Agg")

    def save_all(self) -> list[Path]:
        """Save and close every currently open figure."""
        paths = []
        for figure_number in plt.get_fignums():
            self.counter += 1
            figure = plt.figure(figure_number)
            path = self.output_dir / f"{self.prefix}_{self.counter:03d}.{self.image_format}"
            figure.savefig(path, dpi=self.dpi, bbox_inches="tight")
            paths.append(path)
        plt.close("all")
        return paths
