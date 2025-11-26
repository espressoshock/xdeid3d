"""
Figure generation utilities for evaluation results.

This module provides tools for creating publication-quality
figures from evaluation results, including:

- Metric time series plots
- Score distribution histograms
- Comparison grids
- Performance summaries
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

__all__ = [
    "FigureConfig",
    "FigureGenerator",
    "create_metric_plot",
    "create_distribution_plot",
    "create_comparison_grid",
    "create_summary_figure",
    "save_figure",
]


@dataclass
class FigureConfig:
    """
    Configuration for figure generation.

    Attributes:
        figsize: Figure size in inches (width, height)
        dpi: Resolution in dots per inch
        style: Matplotlib style name
        colormap: Default colormap for heatmaps
        font_size: Base font size
        title_size: Title font size
        label_size: Label font size
    """

    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 150
    style: str = "default"
    colormap: str = "viridis"
    font_size: int = 10
    title_size: int = 12
    label_size: int = 10
    grid_alpha: float = 0.3
    line_width: float = 2.0
    marker_size: float = 6.0


class FigureGenerator:
    """
    Generator for evaluation result figures.

    Provides a unified interface for creating various figure types
    from evaluation data.

    Args:
        config: Figure configuration

    Example:
        >>> gen = FigureGenerator()
        >>> gen.add_metric_series("score", scores, frames)
        >>> fig = gen.create_time_series()
        >>> gen.save("output.png")
    """

    def __init__(self, config: Optional[FigureConfig] = None):
        self.config = config or FigureConfig()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._current_figure = None

    def add_metric_series(
        self,
        name: str,
        values: np.ndarray,
        x_values: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        """
        Add a metric time series.

        Args:
            name: Metric name
            values: Array of metric values
            x_values: Optional x-axis values (default: indices)
            label: Display label (default: name)
            color: Line color
        """
        values = np.asarray(values)
        if x_values is None:
            x_values = np.arange(len(values))
        else:
            x_values = np.asarray(x_values)

        self._data[name] = {
            'values': values,
            'x_values': x_values,
            'label': label or name,
            'color': color,
            'type': 'series',
        }

    def add_distribution(
        self,
        name: str,
        values: np.ndarray,
        bins: int = 30,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        """
        Add values for histogram distribution.

        Args:
            name: Distribution name
            values: Array of values
            bins: Number of histogram bins
            label: Display label
            color: Bar color
        """
        self._data[name] = {
            'values': np.asarray(values),
            'bins': bins,
            'label': label or name,
            'color': color,
            'type': 'distribution',
        }

    def add_scores_with_positions(
        self,
        name: str,
        scores: np.ndarray,
        positions: np.ndarray,
        label: Optional[str] = None,
    ) -> None:
        """
        Add scores with 2D positions (for scatter plots).

        Args:
            name: Data name
            scores: Array of scores
            positions: Nx2 array of (x, y) positions
            label: Display label
        """
        self._data[name] = {
            'scores': np.asarray(scores),
            'positions': np.asarray(positions),
            'label': label or name,
            'type': 'scatter',
        }

    def create_time_series(
        self,
        title: Optional[str] = None,
        xlabel: str = "Frame",
        ylabel: str = "Score",
        show_legend: bool = True,
    ) -> np.ndarray:
        """
        Create time series plot from added data.

        Args:
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_legend: Show legend

        Returns:
            RGB image array
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

            for name, data in self._data.items():
                if data.get('type') != 'series':
                    continue

                ax.plot(
                    data['x_values'],
                    data['values'],
                    label=data['label'],
                    color=data.get('color'),
                    linewidth=self.config.line_width,
                )

            ax.set_xlabel(xlabel, fontsize=self.config.label_size)
            ax.set_ylabel(ylabel, fontsize=self.config.label_size)

            if title:
                ax.set_title(title, fontsize=self.config.title_size)

            if show_legend and len(self._data) > 1:
                ax.legend()

            ax.grid(True, alpha=self.config.grid_alpha)

            # Convert to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            self._current_figure = fig
            plt.close(fig)

            return img

        except ImportError:
            return self._create_fallback_image("Time series plot (matplotlib required)")

    def create_distribution(
        self,
        title: Optional[str] = None,
        xlabel: str = "Value",
        ylabel: str = "Frequency",
    ) -> np.ndarray:
        """
        Create histogram distribution plot.

        Args:
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label

        Returns:
            RGB image array
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

            for name, data in self._data.items():
                if data.get('type') != 'distribution':
                    continue

                ax.hist(
                    data['values'],
                    bins=data.get('bins', 30),
                    label=data['label'],
                    color=data.get('color'),
                    alpha=0.7,
                    edgecolor='black',
                )

            ax.set_xlabel(xlabel, fontsize=self.config.label_size)
            ax.set_ylabel(ylabel, fontsize=self.config.label_size)

            if title:
                ax.set_title(title, fontsize=self.config.title_size)

            ax.grid(True, alpha=self.config.grid_alpha)

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            self._current_figure = fig
            plt.close(fig)

            return img

        except ImportError:
            return self._create_fallback_image("Distribution plot (matplotlib required)")

    def create_scatter(
        self,
        title: Optional[str] = None,
        xlabel: str = "X",
        ylabel: str = "Y",
        colorbar_label: str = "Score",
    ) -> np.ndarray:
        """
        Create scatter plot with color-coded scores.

        Args:
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            colorbar_label: Colorbar label

        Returns:
            RGB image array
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

            for name, data in self._data.items():
                if data.get('type') != 'scatter':
                    continue

                scatter = ax.scatter(
                    data['positions'][:, 0],
                    data['positions'][:, 1],
                    c=data['scores'],
                    cmap=self.config.colormap,
                    s=self.config.marker_size ** 2,
                )

                plt.colorbar(scatter, ax=ax, label=colorbar_label)

            ax.set_xlabel(xlabel, fontsize=self.config.label_size)
            ax.set_ylabel(ylabel, fontsize=self.config.label_size)

            if title:
                ax.set_title(title, fontsize=self.config.title_size)

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            self._current_figure = fig
            plt.close(fig)

            return img

        except ImportError:
            return self._create_fallback_image("Scatter plot (matplotlib required)")

    def save(self, path: Union[str, Path], dpi: Optional[int] = None) -> None:
        """
        Save the most recently created figure.

        Args:
            path: Output file path
            dpi: Override DPI
        """
        if self._current_figure is not None:
            self._current_figure.savefig(
                path,
                dpi=dpi or self.config.dpi,
                bbox_inches='tight',
            )

    def clear(self) -> None:
        """Clear all stored data."""
        self._data.clear()
        self._current_figure = None

    def _create_fallback_image(self, message: str) -> np.ndarray:
        """Create a simple fallback image with text."""
        h = int(self.config.figsize[1] * self.config.dpi)
        w = int(self.config.figsize[0] * self.config.dpi)
        img = np.ones((h, w, 3), dtype=np.uint8) * 240

        # Add simple text (without cv2 dependency)
        # Just return gray image with centered darker region
        center_h = h // 2
        center_w = w // 2
        img[center_h - 20:center_h + 20, center_w - 100:center_w + 100] = 200

        return img


def create_metric_plot(
    values: np.ndarray,
    x_values: Optional[np.ndarray] = None,
    title: str = "Metric Over Time",
    xlabel: str = "Frame",
    ylabel: str = "Score",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 150,
) -> np.ndarray:
    """
    Create a simple metric time series plot.

    Args:
        values: Array of metric values
        x_values: Optional x-axis values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        dpi: Resolution

    Returns:
        RGB image array

    Example:
        >>> scores = np.random.rand(100)
        >>> img = create_metric_plot(scores, title="Identity Score")
    """
    gen = FigureGenerator(FigureConfig(figsize=figsize, dpi=dpi))
    gen.add_metric_series("metric", values, x_values)
    return gen.create_time_series(title=title, xlabel=xlabel, ylabel=ylabel)


def create_distribution_plot(
    values: np.ndarray,
    bins: int = 30,
    title: str = "Score Distribution",
    xlabel: str = "Score",
    ylabel: str = "Frequency",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 150,
) -> np.ndarray:
    """
    Create a histogram distribution plot.

    Args:
        values: Array of values
        bins: Number of bins
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        dpi: Resolution

    Returns:
        RGB image array
    """
    gen = FigureGenerator(FigureConfig(figsize=figsize, dpi=dpi))
    gen.add_distribution("dist", values, bins=bins)
    return gen.create_distribution(title=title, xlabel=xlabel, ylabel=ylabel)


def create_comparison_grid(
    images: Sequence[np.ndarray],
    labels: Optional[Sequence[str]] = None,
    ncols: int = 4,
    cell_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Create a grid of images for comparison.

    Args:
        images: List of images
        labels: Optional labels for each image
        ncols: Number of columns
        cell_size: Size of each cell (width, height)

    Returns:
        Combined grid image

    Example:
        >>> imgs = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(8)]
        >>> grid = create_comparison_grid(imgs, ncols=4)
    """
    if not images:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols

    # Create output grid
    grid_h = nrows * cell_size[1]
    grid_w = ncols * cell_size[0]
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    for i, img in enumerate(images):
        row = i // ncols
        col = i % ncols

        # Resize image if needed
        if img.shape[:2] != (cell_size[1], cell_size[0]):
            img = _resize_image(img, cell_size)

        # Ensure 3 channels
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # Place in grid
        y_start = row * cell_size[1]
        x_start = col * cell_size[0]
        grid[y_start:y_start + cell_size[1], x_start:x_start + cell_size[0]] = img

    # Add labels if provided
    if labels:
        grid = _add_labels_to_grid(grid, labels, ncols, cell_size)

    return grid


def create_summary_figure(
    metrics: Dict[str, float],
    title: str = "Evaluation Summary",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 150,
) -> np.ndarray:
    """
    Create a bar chart summary of metrics.

    Args:
        metrics: Dictionary of metric names to values
        title: Figure title
        figsize: Figure size
        dpi: Resolution

    Returns:
        RGB image array

    Example:
        >>> metrics = {"PSNR": 28.5, "SSIM": 0.92, "Identity": 0.85}
        >>> img = create_summary_figure(metrics)
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        names = list(metrics.keys())
        values = list(metrics.values())

        bars = ax.bar(names, values, color='steelblue', edgecolor='black')

        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{value:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
            )

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return img

    except ImportError:
        h = int(figsize[1] * dpi)
        w = int(figsize[0] * dpi)
        return np.ones((h, w, 3), dtype=np.uint8) * 240


def save_figure(
    image: np.ndarray,
    path: Union[str, Path],
    format: Optional[str] = None,
) -> None:
    """
    Save an image array to file.

    Args:
        image: RGB image array
        path: Output file path
        format: Image format (inferred from extension if not specified)

    Example:
        >>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> save_figure(img, "output.png")
    """
    path = Path(path)

    try:
        import imageio
        imageio.imwrite(str(path), image)
    except ImportError:
        try:
            from PIL import Image
            Image.fromarray(image).save(str(path))
        except ImportError:
            # Fall back to raw numpy save
            np.save(str(path.with_suffix('.npy')), image)


def _resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    """Resize image to target size."""
    try:
        from PIL import Image
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize(size, Image.Resampling.LANCZOS)
        return np.array(pil_img)
    except ImportError:
        # Simple nearest-neighbor resize
        h, w = image.shape[:2]
        target_w, target_h = size

        y_indices = (np.arange(target_h) * h / target_h).astype(int)
        x_indices = (np.arange(target_w) * w / target_w).astype(int)

        return image[np.ix_(y_indices, x_indices)]


def _add_labels_to_grid(
    grid: np.ndarray,
    labels: Sequence[str],
    ncols: int,
    cell_size: Tuple[int, int],
) -> np.ndarray:
    """Add text labels to grid image."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        pil_img = Image.fromarray(grid)
        draw = ImageDraw.Draw(pil_img)

        # Try to get a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            except OSError:
                font = ImageFont.load_default()

        for i, label in enumerate(labels):
            row = i // ncols
            col = i % ncols

            x = col * cell_size[0] + 5
            y = row * cell_size[1] + 5

            # Draw background
            bbox = draw.textbbox((x, y), label, font=font)
            draw.rectangle(
                [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                fill='white'
            )
            draw.text((x, y), label, fill='black', font=font)

        return np.array(pil_img)

    except ImportError:
        return grid
