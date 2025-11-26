# X-DeID3D Examples

This directory contains example scripts demonstrating how to use X-DeID3D.

## Examples

### 01_basic_evaluation.py

Basic evaluation of a single image pair with quality metrics.

```bash
python 01_basic_evaluation.py original.jpg anonymized.jpg
```

### 02_batch_evaluation.py

Batch evaluation of a directory of images using the evaluation pipeline.

```bash
python 02_batch_evaluation.py data/original/ data/anonymized/
```

### 03_generate_heatmap.py

Generate spherical heatmaps from evaluation results.

```bash
# From results file
python 03_generate_heatmap.py results.json

# With synthetic data (demo)
python 03_generate_heatmap.py
```

### 04_custom_metric.py

Create and register a custom evaluation metric.

```bash
python 04_custom_metric.py [original.jpg anonymized.jpg]
```

### 05_custom_anonymizer.py

Create and register a custom anonymizer implementation.

```bash
python 05_custom_anonymizer.py input.jpg [output.jpg]
```

### 06_mesh_visualization.py

Generate colored 3D meshes from evaluation data.

```bash
# From results file
python 06_mesh_visualization.py results.json

# With synthetic data (demo)
python 06_mesh_visualization.py
```

## Running Without Data

Most examples can run without input data using synthetic test data:

```bash
python 03_generate_heatmap.py  # Generates synthetic heatmap
python 04_custom_metric.py     # Uses random images
python 05_custom_anonymizer.py # Uses random image
python 06_mesh_visualization.py # Uses synthetic scores
```

## Prerequisites

Ensure X-DeID3D is installed:

```bash
pip install -e ..
```

For full functionality:

```bash
pip install -e "..[full]"
```

## More Information

See the [documentation](../docs/) for detailed API reference and guides.
