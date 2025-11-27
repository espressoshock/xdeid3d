# X-DeID3D Experiments Viewer

A modern web application for exploring and analyzing X-DeID3D experiments with interactive visualizations, metrics comparison, and 3D heatmap viewing.

## Features

- **Interactive Dashboard**: Browse experiments organized by configuration (Head, FFHQ, Cats) and seed
- **Rich Media Support**: View generated videos, comparison videos, depth maps, and 3D mesh visualizations
- **Metrics Analysis**: Compare performance metrics across experiments with interactive charts
- **Filtering & Sorting**: Filter by configuration, seed range, and sort by various criteria
- **Comparison Mode**: Select multiple experiments to compare side-by-side
- **Responsive Design**: Modern UI that works on desktop and mobile devices

## Quick Start

### Option 1: Using the Launch Script (Recommended)

```bash
cd experiments_viewer
./launch_viewer.sh
```

The script will:
- Create a virtual environment
- Install dependencies
- Start the web server at http://localhost:5001

### Option 2: Manual Setup

```bash
cd experiments_viewer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python backend/app.py
```

## Usage

1. **Open the viewer**: Navigate to http://localhost:5001 in your web browser

2. **Browse Experiments**: 
   - View experiments in grid or list mode
   - Filter by configuration type or seed range
   - Sort by config/seed, date, or metric scores

3. **View Details**: Click on any experiment to see:
   - Full experiment information
   - All generated videos (original, GUARD-processed, comparison, depth, heatmap)
   - Image grids and visualizations
   - Detailed metrics table and charts
   - Downloadable files

4. **Compare Experiments**:
   - Select multiple experiments using checkboxes
   - Click "Compare" button
   - View side-by-side comparisons of metrics, videos, and images

## Directory Structure

```
experiments_viewer/
├── backend/
│   ├── app.py              # Flask application
│   └── utils/
│       └── experiment_reader.py  # Experiment data reader
├── frontend/
│   ├── index.html          # Main HTML file
│   ├── css/
│   │   └── styles.css      # Custom styles
│   └── js/
│       ├── app.js          # Main application logic
│       └── components/     # Reusable UI components
├── requirements.txt        # Python dependencies
├── launch_viewer.sh        # Launch script
└── README.md              # This file
```

## API Endpoints

- `GET /api/experiments` - List all experiments
- `GET /api/experiments/<cfg>/<seed>` - Get specific experiment details
- `GET /api/metrics/summary` - Get metrics summary across all experiments
- `GET /api/comparisons` - Get experiments grouped for comparison
- `GET /api/media/<path>` - Serve media files (videos, images, meshes)
- `GET /api/download/<path>` - Download experiment files

## Requirements

- Python 3.7+
- Flask 2.3+
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Troubleshooting

1. **Port already in use**: If port 5001 is already in use, edit `backend/app.py` and change the port number in the last line.

2. **Experiments not showing**: Make sure your experiments are in the `experiments/` directory at the project root, organized as `experiments/<cfg>/<seed>/`.

3. **Videos not playing**: Ensure your browser supports MP4 video playback. The viewer uses HTML5 video elements.

## Development

To modify the viewer:

1. Backend changes: Edit files in `backend/` and restart the server
2. Frontend changes: Edit files in `frontend/` and refresh the browser
3. No build step required - the application uses vanilla JavaScript

## License

Part of the X-DeID3D project (XAI for de-identification auditing).