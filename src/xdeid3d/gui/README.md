# X-DeID3D GUI Application

Modern web-based GUI for X-DeID3D identity auditing platform.

## Architecture

- **Frontend**: React + Vite + shadcn/ui (dark theme) + TailwindCSS
- **Backend**: FastAPI with WebSocket support
- **Communication**: Real-time WebSocket for progressive preview

## Directory Structure

```
gui/
├── frontend/          # React application
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/              # shadcn components
│   │   │   ├── generation/      # Generation stage components
│   │   │   ├── layout/          # Layout components
│   │   │   └── shared/          # Shared components
│   │   ├── hooks/               # Custom React hooks
│   │   ├── lib/                 # Utilities
│   │   ├── types/               # TypeScript types
│   │   └── pages/               # Page components
│   └── package.json
│
└── backend/           # FastAPI application
    ├── routers/       # API route handlers
    ├── services/      # Business logic (SphereHead, PTI, SD)
    ├── models/        # Pydantic models
    └── utils/         # Utility functions
```

## Development Stages

### Stage 1: Generation (Current)
- Three generation methods: Seed, Upload, Text-to-Image
- Real-time progressive preview
- Quality presets + advanced settings
- 360° 3D rotation preview

### Stage 2: Customization (Future)
- Background alteration
- Feature customization (gaze, smile, expressions)
- Camera pose/rotation with 3D viewer

### Stage 3: Audit & Visualization (Future)
- Video/frame generation
- De-identification network processing
- Metrics computation
- 3D heatmap visualization

## Setup Instructions

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:5173`

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend will be available at `http://localhost:8000`

## Technology Stack

### Frontend
- React 18
- Vite
- TypeScript
- shadcn/ui components
- TailwindCSS
- Three.js (@react-three/fiber)
- Zustand (state management)

### Backend
- FastAPI
- WebSocket (real-time communication)
- PyTorch (SphereHead inference)
- Stable Diffusion (text-to-image)
- PIL/OpenCV (image processing)

## Features

- ✅ Seed-based 3D head generation
- ✅ Image upload with PTI (Pivotal Tuning Inversion)
- ✅ Text-to-image with Stable Diffusion integration
- ✅ Real-time progressive preview
- ✅ 360° rotation preview
- ✅ Quality presets (Fast/Balanced/High Quality)
- ✅ Advanced parameter controls
- ✅ Dark theme UI

## API Endpoints

### REST
- `POST /api/generation/seed` - Generate from seed
- `POST /api/generation/upload` - Upload image for PTI
- `POST /api/generation/text-to-image` - Text-to-image generation
- `GET /api/outputs/{filename}` - Serve generated files

### WebSocket
- `WS /ws/generation/{session_id}` - Real-time progress updates

## License

Same as parent X-DeID3D project.
