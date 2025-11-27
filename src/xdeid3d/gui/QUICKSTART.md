# X-DeID3D GUI - Quick Start Guide

Get the GUI running in 5 minutes!

## 🚀 Quick Start (Development Mode)

### 1. Backend Setup

```bash
# Navigate to backend
cd gui/backend

# Activate existing SphereHead environment
conda activate SphereHead

# Install additional dependencies
pip install fastapi uvicorn websockets python-multipart pydantic

# Create output directories
mkdir -p outputs uploads

# Start backend server
python main.py
```

✅ Backend running at: **http://localhost:8000**

### 2. Frontend Setup

Open a **new terminal**:

```bash
# Navigate to frontend
cd gui/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

✅ Frontend running at: **http://localhost:5001**

### 3. Open in Browser

Navigate to: **http://localhost:5173**

You should see the X-DeID3D GUI with three generation tabs!

---

## 📁 What You'll See

### Generation Tab (Current)

Three methods to generate 3D heads:

1. **🎲 Seed Generator**
   - Enter seed number (0-999999)
   - Select quality preset (Fast/Balanced/High Quality)
   - Optional: Expand advanced settings
   - Click "Generate 3D Head"

2. **📤 Image Upload**
   - Drag & drop or click to upload face image
   - Supports PNG, JPG (max 10MB)
   - Runs PTI (Pivotal Tuning Inversion)
   - Takes ~5 minutes

3. **✍️ Text Prompt**
   - Describe the face you want
   - Adjust Stable Diffusion parameters
   - Click example prompts for ideas
   - Generates image first, then converts to 3D

### Customization Tab (Coming Soon)
Currently shows placeholder - Stage 2 development

### Audit Tab (Coming Soon)
Currently shows placeholder - Stage 3 development

---

## 🔧 Current Limitations (MVP Stage)

⚠️ **This is the initial UI implementation!**

- **Backend services are stubs**: Actual generation not yet connected
- **Progress updates**: Simulated (not real progress)
- **3D preview**: Not yet implemented
- **Download**: Not yet functional

### What Works Now

✅ UI/UX and layout
✅ Dark theme
✅ WebSocket connection
✅ State management
✅ File upload
✅ Form validation

### What's Next

🔄 Connect to actual SphereHead core
🔄 Real PTI implementation
🔄 Stable Diffusion integration
🔄 360° 3D viewer with Three.js
🔄 Download/export functionality

---

## 🐛 Troubleshooting

### Backend won't start

```bash
# Check Python dependencies
conda activate SphereHead
pip list | grep fastapi

# Install if missing
pip install fastapi uvicorn
```

### Frontend won't start

```bash
# Clear cache and reinstall
cd gui/frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Port already in use

```bash
# Backend (port 8000)
lsof -i :8000
kill -9 <PID>

# Frontend (port 5173)
lsof -i :5173
kill -9 <PID>
```

### WebSocket connection fails

1. Ensure backend is running: `curl http://localhost:8000/health`
2. Check browser console for errors
3. Verify CORS settings in `backend/main.py`

---

## 📖 Next Steps

1. **Try the UI**: Explore all three generation methods
2. **Read Full Guide**: See `DEVELOPMENT.md` for detailed setup
3. **Implement Backend**: Connect services to SphereHead core
4. **Add 3D Preview**: Implement Three.js viewer

---

## 💡 Tips

- Use **Fast** preset for quick testing (lower quality)
- Use **High Quality** preset for final generation (slower)
- Advanced settings override quality presets
- WebSocket connection status shown at bottom left

---

## 📸 Expected UI

When you open the app, you should see:

- **Top**: X-DeID3D header with logo
- **Tabs**: Generation / Customization (disabled) / Audit (disabled)
- **Left Panel** (400px): Generation method tabs and controls
- **Right Panel**: Preview area with progress tracking
- **Bottom Left**: WebSocket connection indicator (green = connected)

---


