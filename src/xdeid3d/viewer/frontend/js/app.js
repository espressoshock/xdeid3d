// Compact Experiments Viewer Application
class CompactExperimentsViewer {
    constructor() {
        this.experiments = [];
        this.filteredExperiments = [];
        this.selectedExperiment = null;
        this.selectedConfig = null;
        this.tileSize = 'medium';
        this.viewMode = 'grid';
        this.videoThumbnails = new Map();
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadExperiments();
    }
    
    initializeElements() {
        // Sidebar
        this.sidebar = document.getElementById('sidebar');
        this.sidebarToggle = document.getElementById('sidebarToggle');
        this.configFilters = document.getElementById('configFilters');
        this.seedMin = document.getElementById('seedMin');
        this.seedMax = document.getElementById('seedMax');
        this.totalCount = document.getElementById('totalCount');
        this.refreshBtn = document.getElementById('refreshBtn');
        
        // Main content
        this.viewTitle = document.getElementById('viewTitle');
        this.filterBadge = document.getElementById('filterBadge');
        this.tilesContainer = document.getElementById('tilesContainer');
        this.loadingState = document.getElementById('loadingState');
        this.emptyState = document.getElementById('emptyState');
        this.experimentGrid = document.getElementById('experimentGrid');
        
        // Detail panel
        this.detailPanel = document.getElementById('detailPanel');
        this.detailTitle = document.getElementById('detailTitle');
        this.detailContent = document.getElementById('detailContent');
        this.closeDetailBtn = document.getElementById('closeDetailBtn');
        
        // View buttons
        this.gridViewBtn = document.getElementById('gridViewBtn');
        this.listViewBtn = document.getElementById('listViewBtn');
        
        // Tile size buttons
        this.tileSizeButtons = document.querySelectorAll('.tile-size-btn');
        
        // Interactive Analysis elements
        this.interactiveAnalysis = document.getElementById('interactiveAnalysis');
        this.closeAnalysisBtn = document.getElementById('closeAnalysisBtn');
        this.analysisVideo = document.getElementById('analysisVideo');
        this.videoWrapper = document.getElementById('videoWrapper');
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.prevFrameBtn = document.getElementById('prevFrameBtn');
        this.nextFrameBtn = document.getElementById('nextFrameBtn');
        this.videoSeeker = document.getElementById('videoSeeker');
        this.currentTime = document.getElementById('currentTime');
        this.duration = document.getElementById('duration');
        this.currentFrame = document.getElementById('currentFrame');
        this.totalFrames = document.getElementById('totalFrames');
        this.fullscreenBtn = document.getElementById('fullscreenBtn');
        this.playbackSpeed = document.getElementById('playbackSpeed');
        this.videoOptions = document.getElementById('videoOptions');
        this.analysisInfo = document.getElementById('analysisInfo');
        this.analysisSubtitle = document.getElementById('analysisSubtitle');
        this.loopVideo = document.getElementById('loopVideo');
        this.videoProgressFill = document.getElementById('videoProgressFill');
        
        // 3D viewer elements
        this.videoContainer = document.getElementById('videoContainer');
        this.threeDContainer = document.getElementById('threeDContainer');
        this.threeDViewer = document.getElementById('threeDViewer');
        this.plySelector = document.getElementById('plySelector');
        this.videoControls = document.getElementById('videoControls');
        
        // Metric info elements
        this.metricInfoBtn = document.getElementById('metricInfoBtn');
        this.metricInfoPanel = document.getElementById('metricInfoPanel');
        this.closeMetricInfo = document.getElementById('closeMetricInfo');
        this.metricName = document.getElementById('metricName');
        this.metricCategory = document.getElementById('metricCategory');
        this.metricDescription = document.getElementById('metricDescription');
        this.metricInterpretation = document.getElementById('metricInterpretation');
        this.metricDetails = document.getElementById('metricDetails');
        
        // Colorbar elements
        this.colorbar = document.getElementById('colorbar');
        this.colorbarCanvas = document.getElementById('colorbarCanvas');
        this.colorbarMax = document.getElementById('colorbarMax');
        this.colorbarMid = document.getElementById('colorbarMid');
        this.colorbarMin = document.getElementById('colorbarMin');
        this.colorbarHotMeaning = document.getElementById('colorbarHotMeaning');
        this.colorbarColdMeaning = document.getElementById('colorbarColdMeaning');
        
        this.currentAnalysisExp = null;
        this.currentVideoType = null;
        this.frameRate = 30; // Assume 30fps, will update when video loads
        this.isDragging = false;
        this.currentMetric = null;
        
        // Three.js viewer
        this.threeViewer = null;
        
        // Setup visibility observer for interactive analysis
        this.setupVisibilityObserver();
    }
    
    attachEventListeners() {
        // Sidebar toggle
        this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        
        // Filters
        this.seedMin.addEventListener('input', () => this.applyFilters());
        this.seedMax.addEventListener('input', () => this.applyFilters());
        
        // Refresh
        this.refreshBtn.addEventListener('click', () => this.loadExperiments());
        
        // Detail panel
        this.closeDetailBtn.addEventListener('click', () => this.closeDetail());
        
        // View mode
        this.gridViewBtn.addEventListener('click', () => this.setViewMode('grid'));
        this.listViewBtn.addEventListener('click', () => this.setViewMode('list'));
        
        // Tile size
        this.tileSizeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => this.setTileSize(e.target.dataset.size));
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (!this.metricInfoPanel.classList.contains('hidden')) {
                    this.metricInfoPanel.classList.add('hidden');
                } else if (!this.interactiveAnalysis.classList.contains('hidden')) {
                    this.closeInteractiveAnalysis();
                } else {
                    this.closeDetail();
                }
            } else if (e.key === 'i' || e.key === 'I') {
                // Toggle metric info with 'i' key
                if (!this.threeDContainer.classList.contains('hidden') && this.currentMetric) {
                    if (this.metricInfoPanel.classList.contains('hidden')) {
                        this.showMetricInfo();
                    } else {
                        this.metricInfoPanel.classList.add('hidden');
                    }
                }
            }
        });
        
        // Interactive Analysis event listeners
        this.closeAnalysisBtn.addEventListener('click', () => this.closeInteractiveAnalysis());
        this.playPauseBtn.addEventListener('click', () => this.togglePlayPause());
        this.fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        this.prevFrameBtn.addEventListener('click', () => this.stepFrame(-1));
        this.nextFrameBtn.addEventListener('click', () => this.stepFrame(1));
        
        // Video controls
        this.videoSeeker.addEventListener('mousedown', () => {
            this.isDragging = true;
        });
        
        this.videoSeeker.addEventListener('mouseup', () => {
            this.isDragging = false;
        });
        
        this.videoSeeker.addEventListener('input', (e) => {
            const progress = e.target.value;
            const time = (this.analysisVideo.duration * progress) / 100;
            this.analysisVideo.currentTime = time;
            // Update progress fill immediately
            if (this.videoProgressFill) {
                this.videoProgressFill.style.width = `${progress}%`;
            }
        });
        
        // Attach video event listeners
        this.attachVideoEventListeners();
        
        this.playbackSpeed.addEventListener('change', (e) => {
            this.analysisVideo.playbackRate = parseFloat(e.target.value);
        });
        
        // Loop checkbox
        this.loopVideo.addEventListener('change', (e) => {
            this.analysisVideo.loop = e.target.checked;
        });
        
        // PLY selector
        this.plySelector.addEventListener('change', (e) => {
            if (this.threeViewer) {
                this.loadPLYModel(e.target.value);
                this.updateMetricInfo(e.target.value);
                const metric = e.target.value.match(/mesh_(.+)\.ply$/)?.[1] || '';
                this.updateColorbarLabels(metric);
                // Automatically show metric info when selection changes
                this.showMetricInfo();
            }
        });
        
        // Metric info button
        this.metricInfoBtn.addEventListener('click', () => {
            this.showMetricInfo();
        });
        
        // Close metric info
        this.closeMetricInfo.addEventListener('click', () => {
            this.metricInfoPanel.classList.add('hidden');
        });
        
        // Resize video when window resizes
        window.addEventListener('resize', () => {
            if (!this.interactiveAnalysis.classList.contains('hidden')) {
                if (!this.threeDContainer.classList.contains('hidden')) {
                    this.resizeThreeViewer();
                } else {
                    this.resizeVideo();
                }
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Escape key to close detail panel
            if (e.key === 'Escape') {
                if (!this.detailPanel.classList.contains('hidden')) {
                    this.hideDetail();
                } else if (!this.interactiveAnalysis.classList.contains('hidden')) {
                    this.closeInteractiveAnalysis();
                }
            }
        });
    }
    
    toggleSidebar() {
        const isCollapsed = this.sidebar.classList.contains('collapsed');
        
        if (isCollapsed) {
            this.sidebar.classList.remove('collapsed');
            this.sidebar.classList.add('expanded');
        } else {
            this.sidebar.classList.remove('expanded');
            this.sidebar.classList.add('collapsed');
        }
    }
    
    async loadExperiments() {
        this.showLoading(true);
        
        try {
            const response = await fetch('/api/experiments');
            const data = await response.json();
            
            this.experiments = data.experiments;
            this.filteredExperiments = [...this.experiments];
            
            this.updateStats();
            this.createConfigFilters();
            this.applyFilters();
            
            // Pre-generate video thumbnails
            this.generateVideoThumbnails();
            
        } catch (error) {
            console.error('Failed to load experiments:', error);
            this.showError('Failed to load experiments');
        } finally {
            this.showLoading(false);
        }
    }
    
    updateStats() {
        this.totalCount.textContent = this.experiments.length;
        
        // Count by configuration
        const configCounts = {};
        this.experiments.forEach(exp => {
            configCounts[exp.cfg] = (configCounts[exp.cfg] || 0) + 1;
        });
        
        return configCounts;
    }
    
    createConfigFilters() {
        const configCounts = this.updateStats();
        
        // Add "All" button
        let filtersHTML = `
            <button class="config-filter-btn w-full text-left px-3 py-2 rounded hover:bg-gray-100 transition-colors flex items-center justify-between sidebar-item ${!this.selectedConfig ? 'bg-indigo-100 text-indigo-700' : ''}" 
                    data-config="" title="All Configurations">
                <div class="flex items-center">
                    <i class="fas fa-border-all sidebar-icon"></i>
                    <span class="text-sm font-medium sidebar-text ml-3">All</span>
                </div>
                <span class="text-xs bg-gray-200 px-2 py-1 rounded-full sidebar-count">${this.experiments.length}</span>
            </button>
        `;
        
        // Add config-specific buttons
        Object.entries(configCounts).forEach(([cfg, count]) => {
            // Map configuration names for display
            const displayName = this.getConfigDisplayName(cfg);
            const icon = this.getConfigIcon(cfg);
            const isSelected = this.selectedConfig === cfg;
            
            filtersHTML += `
                <button class="config-filter-btn w-full text-left px-3 py-2 rounded hover:bg-gray-100 transition-colors flex items-center justify-between sidebar-item ${isSelected ? 'bg-indigo-100 text-indigo-700' : ''}" 
                        data-config="${cfg}" title="${cfg}">
                    <div class="flex items-center">
                        <i class="fas ${icon} sidebar-icon"></i>
                        <span class="text-sm font-medium sidebar-text ml-3">${displayName}</span>
                    </div>
                    <span class="text-xs bg-gray-200 px-2 py-1 rounded-full sidebar-count">${count}</span>
                </button>
            `;
        });
        
        this.configFilters.innerHTML = filtersHTML;
        
        // Attach click handlers
        this.configFilters.querySelectorAll('.config-filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const config = btn.dataset.config;
                this.selectedConfig = config || null;
                this.updateConfigSelection();
                this.applyFilters();
            });
        });
    }
    
    updateConfigSelection() {
        this.configFilters.querySelectorAll('.config-filter-btn').forEach(btn => {
            const isSelected = btn.dataset.config === (this.selectedConfig || '');
            btn.classList.toggle('bg-indigo-100', isSelected);
            btn.classList.toggle('text-indigo-700', isSelected);
        });
    }
    
    applyFilters() {
        const minSeed = parseInt(this.seedMin.value) || -Infinity;
        const maxSeed = parseInt(this.seedMax.value) || Infinity;
        
        this.filteredExperiments = this.experiments.filter(exp => {
            if (this.selectedConfig && exp.cfg !== this.selectedConfig) return false;
            if (exp.seed < minSeed || exp.seed > maxSeed) return false;
            return true;
        });
        
        // Update view title and badge
        if (this.selectedConfig) {
            this.viewTitle.textContent = `${this.getConfigDisplayName(this.selectedConfig)} Experiments`;
        } else {
            this.viewTitle.textContent = 'All Experiments';
        }
        
        this.filterBadge.textContent = `${this.filteredExperiments.length} results`;
        
        this.renderExperiments();
    }
    
    setTileSize(size) {
        this.tileSize = size;
        
        // Update button states
        this.tileSizeButtons.forEach(btn => {
            const isActive = btn.dataset.size === size;
            btn.classList.toggle('bg-indigo-600', isActive);
            btn.classList.toggle('text-white', isActive);
            btn.classList.toggle('bg-gray-100', !isActive);
        });
        
        // Update grid columns
        const gridClasses = {
            'small': 'grid-cols-6',
            'medium': 'grid-cols-4',
            'large': 'grid-cols-3',
            'xl': 'grid-cols-2'
        };
        
        this.tilesContainer.className = `grid gap-4 ${gridClasses[size]} max-w-7xl mx-auto transition-all duration-300`;
        this.renderExperiments();
    }
    
    setViewMode(mode) {
        this.viewMode = mode;
        
        if (mode === 'grid') {
            this.gridViewBtn.classList.add('text-gray-600');
            this.gridViewBtn.classList.remove('text-gray-400');
            this.listViewBtn.classList.add('text-gray-400');
            this.listViewBtn.classList.remove('text-gray-600');
        } else {
            this.listViewBtn.classList.add('text-gray-600');
            this.listViewBtn.classList.remove('text-gray-400');
            this.gridViewBtn.classList.add('text-gray-400');
            this.gridViewBtn.classList.remove('text-gray-600');
        }
        
        this.renderExperiments();
    }
    
    renderExperiments() {
        if (this.filteredExperiments.length === 0) {
            this.tilesContainer.innerHTML = '';
            this.emptyState.classList.remove('hidden');
            return;
        }
        
        this.emptyState.classList.add('hidden');
        
        if (this.viewMode === 'grid') {
            this.renderGrid();
        } else {
            this.renderList();
        }
    }
    
    renderGrid() {
        // Reset grid classes based on current tile size
        const gridClasses = {
            'small': 'grid-cols-6',
            'medium': 'grid-cols-4',
            'large': 'grid-cols-3',
            'xl': 'grid-cols-2'
        };
        this.tilesContainer.className = `grid gap-4 ${gridClasses[this.tileSize]} max-w-7xl mx-auto transition-all duration-300`;
        
        const tilesHTML = this.filteredExperiments.map(exp => this.createTile(exp)).join('');
        this.tilesContainer.innerHTML = tilesHTML;
        
        // Attach click handlers and video load handlers
        this.tilesContainer.querySelectorAll('.experiment-tile').forEach((tile, index) => {
            const exp = this.filteredExperiments[index];
            tile.addEventListener('click', () => this.selectExperiment(exp));
            
            // Handle video loading
            const video = tile.querySelector('video');
            if (video) {
                // Show video preview on hover
                let playTimeout;
                tile.addEventListener('mouseenter', () => {
                    playTimeout = setTimeout(() => {
                        video.play().catch(() => {});
                    }, 200);
                });
                
                tile.addEventListener('mouseleave', () => {
                    clearTimeout(playTimeout);
                    video.pause();
                    video.currentTime = 0.1;
                });
                
                // When video metadata is loaded, hide skeleton and show video
                video.addEventListener('loadedmetadata', () => {
                    const skeleton = tile.querySelector('.video-skeleton');
                    if (skeleton) {
                        skeleton.classList.add('hidden');
                    }
                    video.classList.remove('opacity-0');
                });
                
                // Handle video errors
                video.addEventListener('error', () => {
                    const skeleton = tile.querySelector('.video-skeleton');
                    if (skeleton) {
                        skeleton.innerHTML = `
                            <div class="w-full h-full flex items-center justify-center bg-gray-100">
                                <i class="fas fa-exclamation-triangle text-2xl text-gray-400"></i>
                            </div>
                        `;
                    }
                });
            }
        });
    }
    
    renderList() {
        // Reset to list layout
        this.tilesContainer.className = 'space-y-2';
        const listHTML = this.filteredExperiments.map(exp => this.createListItem(exp)).join('');
        this.tilesContainer.innerHTML = listHTML;
        
        // Attach click handlers and video load handlers for list view
        this.tilesContainer.querySelectorAll('.experiment-list-item').forEach((item, index) => {
            const exp = this.filteredExperiments[index];
            item.addEventListener('click', () => this.selectExperiment(exp));
            
            // Handle video loading in list view
            const video = item.querySelector('video');
            if (video) {
                video.addEventListener('loadedmetadata', () => {
                    const skeleton = item.querySelector('.list-video-skeleton');
                    if (skeleton) {
                        skeleton.classList.add('hidden');
                    }
                    video.classList.remove('opacity-0');
                });
                
                video.addEventListener('error', () => {
                    const skeleton = item.querySelector('.list-video-skeleton');
                    if (skeleton) {
                        skeleton.innerHTML = `
                            <div class="w-full h-full flex items-center justify-center bg-gray-200">
                                <i class="fas fa-exclamation-triangle text-gray-400"></i>
                            </div>
                        `;
                    }
                });
            }
        });
    }
    
    createTile(experiment) {
        const videoPath = experiment.files.original_video;
        const hasVideo = !!videoPath;
        
        // Calculate average metric for color coding
        const metrics = Object.values(experiment.metrics || {});
        const avgMetric = metrics.length > 0 ? metrics.reduce((a, b) => a + b) / metrics.length : null;
        
        // Make tiles square using aspect-ratio
        const tileSizes = {
            'small': 'w-full aspect-square',
            'medium': 'w-full aspect-square',
            'large': 'w-full aspect-square',
            'xl': 'w-full aspect-square'
        };
        
        return `
            <div class="experiment-tile relative bg-white rounded-lg overflow-hidden shadow-md transition-all duration-300 ${tileSizes[this.tileSize]} ${this.selectedExperiment?.seed === experiment.seed ? 'selected ring-2 ring-indigo-500' : ''}" data-experiment-seed="${experiment.seed}">
                ${hasVideo ? `
                    <!-- Skeleton loader -->
                    <div class="video-skeleton absolute inset-0 bg-gray-100">
                        <div class="skeleton w-full h-full"></div>
                    </div>
                    <!-- Video element -->
                    <video class="absolute inset-0 w-full h-full object-cover video-preview opacity-0 transition-opacity duration-300" 
                           muted loop preload="metadata" data-src="/api/media/${videoPath}" data-tile-seed="${experiment.seed}">
                        <source src="/api/media/${videoPath}#t=0.1" type="video/mp4">
                    </video>
                ` : `
                    <div class="absolute inset-0 w-full h-full bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center">
                        <i class="fas fa-cube text-3xl text-gray-400"></i>
                    </div>
                `}
                
                <!-- Overlay Info -->
                <div class="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent pointer-events-none"></div>
                
                <div class="absolute bottom-0 left-0 right-0 p-2 text-white">
                    <div class="flex items-center justify-between">
                        <div>
                            <div class="text-xs font-semibold">${this.getConfigDisplayName(experiment.cfg)}</div>
                            <div class="text-xs opacity-80">Seed ${experiment.seed}</div>
                        </div>
                        ${avgMetric !== null ? `
                            <div class="flex space-x-1">
                                ${this.createMetricDots(experiment.metrics)}
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    createListItem(experiment) {
        const videoPath = experiment.files.original_video;
        
        return `
            <div class="experiment-list-item bg-white rounded-lg shadow-sm p-3 hover:shadow-md transition-all duration-300 cursor-pointer flex items-center space-x-3 ${this.selectedExperiment?.seed === experiment.seed ? 'ring-2 ring-indigo-500' : ''}" data-experiment-seed="${experiment.seed}">
                <div class="w-16 h-16 rounded overflow-hidden bg-gray-100 flex-shrink-0 relative">
                    ${videoPath ? `
                        <!-- Skeleton loader for list view -->
                        <div class="list-video-skeleton absolute inset-0 bg-gray-200">
                            <div class="skeleton w-full h-full"></div>
                        </div>
                        <video class="w-full h-full object-cover opacity-0 transition-opacity duration-300" 
                               muted preload="metadata" data-list-seed="${experiment.seed}">
                            <source src="/api/media/${videoPath}#t=0.1" type="video/mp4">
                        </video>
                    ` : `
                        <div class="w-full h-full flex items-center justify-center">
                            <i class="fas fa-cube text-gray-400"></i>
                        </div>
                    `}
                </div>
                
                <div class="flex-1">
                    <div class="font-medium">${this.getConfigDisplayName(experiment.cfg)} - Seed ${experiment.seed}</div>
                    <div class="text-sm text-gray-500">${experiment.metadata.num_frames || 0} frames</div>
                </div>
                
                <div class="flex space-x-1">
                    ${this.createMetricDots(experiment.metrics)}
                </div>
            </div>
        `;
    }
    
    createMetricDots(metrics) {
        if (!metrics || Object.keys(metrics).length === 0) return '';
        
        // Show first 3 metrics as dots
        return Object.entries(metrics).slice(0, 3).map(([key, value]) => {
            const quality = this.getMetricQuality(key, value);
            return `<div class="metric-dot metric-${quality}" title="${key}: ${value.toFixed(3)}"></div>`;
        }).join('');
    }
    
    getMetricQuality(metric, value) {
        const metricLower = metric.toLowerCase();
        
        if (metricLower.includes('distance') || metricLower.includes('mse')) {
            if (value < 0.2) return 'good';
            if (value < 0.4) return 'medium';
            return 'poor';
        }
        
        if (metricLower.includes('similarity') || metricLower.includes('consistency')) {
            if (value > 0.8) return 'good';
            if (value > 0.6) return 'medium';
            return 'poor';
        }
        
        // Default
        if (value > 0.7) return 'good';
        if (value > 0.4) return 'medium';
        return 'poor';
    }
    
    selectExperiment(experiment) {
        this.selectedExperiment = experiment;
        
        // Update selection state without re-rendering
        this.updateSelectionState(experiment);
        
        // Show detail panel with smooth transition
        this.showDetail(experiment);
    }
    
    updateSelectionState(experiment) {
        // Remove all previous selections
        this.tilesContainer.querySelectorAll('.selected, .ring-2').forEach(el => {
            el.classList.remove('selected', 'ring-2', 'ring-indigo-500');
        });
        
        // Find and select the matching tile/item
        const allItems = this.tilesContainer.querySelectorAll('[data-experiment-seed]');
        allItems.forEach((item) => {
            if (item.dataset.experimentSeed == experiment.seed) {
                item.classList.add('selected', 'ring-2', 'ring-indigo-500');
            }
        });
    }
    
    showDetail(experiment) {
        // Show backdrop
        if (!document.querySelector('.detail-backdrop')) {
            const backdrop = document.createElement('div');
            backdrop.className = 'detail-backdrop opacity-0';
            backdrop.addEventListener('click', () => this.hideDetail());
            document.body.appendChild(backdrop);
            
            // Force reflow to enable transition
            setTimeout(() => {
                backdrop.classList.add('opacity-100');
            }, 10);
        }
        
        // Update title with formatted config name
        this.detailTitle.textContent = `${this.getConfigDisplayName(experiment.cfg)} - Seed ${experiment.seed}`;
        
        // Generate detail content
        this.detailContent.innerHTML = this.generateDetailContent(experiment);
        
        // Show panel with slide-in animation
        this.detailPanel.classList.remove('hidden');
        setTimeout(() => {
            this.detailPanel.classList.add('show');
        }, 10);
        
        // Initialize video players
        const videos = this.detailContent.querySelectorAll('video');
        videos.forEach(video => {
            video.addEventListener('loadedmetadata', () => {
                video.play().catch(() => {});
            });
        });
    }
    
    generateDetailContent(experiment) {
        const videos = [];
        
        // Collect all videos
        if (experiment.files.mesh_video) {
            videos.push({ title: '3D Mesh Visualization', path: experiment.files.mesh_video });
        }
        if (experiment.files.mesh_video_compact) {
            videos.push({ title: 'Compact Visualization', path: experiment.files.mesh_video_compact });
        }
        
        return `
            <!-- Info Section -->
            <div class="mb-6">
                <h4 class="text-sm font-semibold text-gray-700 mb-2">Information</h4>
                <dl class="space-y-1 text-sm">
                    <div class="flex justify-between">
                        <dt class="text-gray-500">Configuration</dt>
                        <dd class="font-medium">${this.getConfigDisplayName(experiment.cfg)}</dd>
                    </div>
                    <div class="flex justify-between">
                        <dt class="text-gray-500">Seed</dt>
                        <dd class="font-medium">${experiment.seed}</dd>
                    </div>
                    <div class="flex justify-between">
                        <dt class="text-gray-500">Frames</dt>
                        <dd class="font-medium">${experiment.metadata.num_frames || 'N/A'}</dd>
                    </div>
                    <div class="flex justify-between">
                        <dt class="text-gray-500">Voxel Resolution</dt>
                        <dd class="font-medium">${experiment.metadata.voxel_resolution || 'N/A'}</dd>
                    </div>
                </dl>
            </div>
            
            <!-- Interactive Analysis Button -->
            <div class="mb-6">
                <button onclick="window.app.openInteractiveAnalysis()" class="w-full py-2 px-4 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors text-sm flex items-center justify-center">
                    <i class="fas fa-play-circle mr-2"></i>
                    Open Interactive Analysis
                </button>
            </div>
            
            <!-- Videos Section -->
            ${videos.length > 0 ? `
                <div class="mb-6">
                    <h4 class="text-sm font-semibold text-gray-700 mb-2">Videos</h4>
                    <div class="space-y-3">
                        ${videos.map(video => `
                            <div>
                                <h5 class="text-xs font-medium text-gray-600 mb-1">${video.title}</h5>
                                <video controls class="w-full rounded bg-black" style="max-height: 200px;">
                                    <source src="/api/media/${video.path}" type="video/mp4">
                                </video>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            <!-- Metrics Section -->
            ${Object.keys(experiment.metrics || {}).length > 0 ? `
                <div class="mb-6">
                    <h4 class="text-sm font-semibold text-gray-700 mb-2">Metrics</h4>
                    <div class="space-y-2">
                        ${Object.entries(experiment.metrics).map(([key, value]) => `
                            <div class="flex items-center justify-between p-2 bg-gray-50 rounded">
                                <span class="text-sm text-gray-600">${this.formatMetricName(key)}</span>
                                <span class="text-sm font-medium ${this.getMetricColorClass(key, value)}">${value.toFixed(4)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            <!-- Files Section -->
            <div class="mb-6">
                <h4 class="text-sm font-semibold text-gray-700 mb-2">Files</h4>
                <div class="space-y-1">
                    ${Object.entries(experiment.files).filter(([key]) => key !== 'mesh_files').map(([key, path]) => {
                        if (!path) return '';
                        const filename = path.split('/').pop();
                        return `
                            <a href="/api/download/${path}" 
                               class="flex items-center justify-between p-2 text-sm hover:bg-gray-50 rounded transition-colors">
                                <span class="text-gray-600 truncate">${filename}</span>
                                <i class="fas fa-download text-gray-400"></i>
                            </a>
                        `;
                    }).join('')}
                    
                    ${experiment.files.mesh_files ? experiment.files.mesh_files.map(path => {
                        const filename = path.split('/').pop();
                        return `
                            <a href="/api/download/${path}" 
                               class="flex items-center justify-between p-2 text-sm hover:bg-gray-50 rounded transition-colors">
                                <span class="text-gray-600 truncate">${filename}</span>
                                <i class="fas fa-download text-gray-400"></i>
                            </a>
                        `;
                    }).join('') : ''}
                </div>
            </div>
        `;
    }
    
    formatMetricName(metric) {
        return metric
            .replace(/_/g, ' ')
            .replace(/\b\w/g, char => char.toUpperCase())
            .replace('Arcface', 'ArcFace')
            .replace('Lpips', 'LPIPS')
            .replace('Mse', 'MSE')
            .replace('Psnr', 'PSNR');
    }
    
    getMetricColorClass(metric, value) {
        const quality = this.getMetricQuality(metric, value);
        const colorMap = {
            'good': 'text-green-600',
            'medium': 'text-yellow-600',
            'poor': 'text-red-600'
        };
        return colorMap[quality] || 'text-gray-600';
    }
    
    closeDetail() {
        this.hideDetail();
    }
    
    hideDetail() {
        // Hide panel with slide-out animation
        this.detailPanel.classList.remove('show');
        this.detailPanel.classList.add('translate-x-full');
        
        // Remove backdrop
        const backdrop = document.querySelector('.detail-backdrop');
        if (backdrop) {
            backdrop.classList.remove('opacity-100');
            backdrop.classList.add('opacity-0');
            setTimeout(() => backdrop.remove(), 300);
        }
        
        // Hide panel after animation
        setTimeout(() => {
            this.detailPanel.classList.add('hidden');
        }, 300);
        
        // Clear selection without re-rendering
        this.selectedExperiment = null;
        this.updateSelectionState({ seed: null });
    }
    
    generateVideoThumbnails() {
        // Auto-play videos on hover
        this.tilesContainer.addEventListener('mouseover', (e) => {
            const video = e.target.closest('.experiment-tile')?.querySelector('video');
            if (video && !video.playing) {
                video.play().catch(() => {});
            }
        });
        
        this.tilesContainer.addEventListener('mouseout', (e) => {
            const video = e.target.closest('.experiment-tile')?.querySelector('video');
            if (video) {
                video.pause();
                video.currentTime = 0.1;
            }
        });
    }
    
    showLoading(show) {
        if (show) {
            this.loadingState.classList.remove('hidden');
            this.tilesContainer.innerHTML = '';
        } else {
            this.loadingState.classList.add('hidden');
        }
    }
    
    showError(message) {
        this.tilesContainer.innerHTML = `
            <div class="col-span-full text-center py-12">
                <i class="fas fa-exclamation-triangle text-4xl text-red-500 mb-4"></i>
                <p class="text-gray-600">${message}</p>
            </div>
        `;
    }
    openInteractiveAnalysis() {
        if (!this.selectedExperiment) return;
        
        this.currentAnalysisExp = this.selectedExperiment;
        this.interactiveAnalysis.classList.remove('hidden');
        
        // Force resize after modal is visible
        setTimeout(() => {
            if (!this.threeDContainer.classList.contains('hidden')) {
                this.resizeThreeViewer();
            } else {
                this.resizeVideo();
            }
        }, 50);
        
        // Collect available videos in the correct order
        const videos = [];
        
        // 1. Check for anonymized video first
        if (this.currentAnalysisExp.files.anonymized_video) {
            videos.push({
                type: 'anonymized',
                path: this.currentAnalysisExp.files.anonymized_video,
                title: 'Anonymized Video',
                description: 'GUARD-processed anonymized video'
            });
        }
        
        // 2. Then compact 3D mesh
        if (this.currentAnalysisExp.files.mesh_video_compact) {
            videos.push({
                type: 'mesh_compact',
                path: this.currentAnalysisExp.files.mesh_video_compact,
                title: 'Compact 3D Mesh',
                description: 'Compact mesh visualization'
            });
        }
        
        // 3. Full 3D mesh visualization
        if (this.currentAnalysisExp.files.mesh_video) {
            videos.push({
                type: 'mesh',
                path: this.currentAnalysisExp.files.mesh_video,
                title: '3D Mesh Visualization',
                description: 'Full 3D mesh with all metrics'
            });
        }
        
        // 4. Interactive 3D PLY models
        if (this.currentAnalysisExp.files.mesh_files && this.currentAnalysisExp.files.mesh_files.length > 0) {
            videos.push({
                type: '3d_ply',
                title: '3D Interactive Models',
                description: 'Rotate, pan, and zoom 3D mesh models',
                meshFiles: this.currentAnalysisExp.files.mesh_files
            });
        }
        
        // Update subtitle
        this.analysisSubtitle.textContent = `${this.getConfigDisplayName(this.currentAnalysisExp.cfg)} - Seed ${this.currentAnalysisExp.seed}`;
        
        // Render video options
        this.videoOptions.innerHTML = videos.map((video, index) => `
            <label class="video-option ${index === 0 ? 'selected' : ''}">
                <input type="radio" name="videoType" value="${video.type}" ${index === 0 ? 'checked' : ''}>
                <span class="radio-dot"></span>
                <div>
                    <div class="font-medium text-gray-800">${video.title}</div>
                    <div class="text-xs text-gray-500 mt-1">${video.description}</div>
                </div>
            </label>
        `).join('');
        
        // Add event listeners to radio buttons
        this.videoOptions.querySelectorAll('input[name="videoType"]').forEach(input => {
            input.addEventListener('change', (e) => {
                // Update selected state
                this.videoOptions.querySelectorAll('.video-option').forEach(opt => {
                    opt.classList.remove('selected');
                });
                e.target.closest('.video-option').classList.add('selected');
                
                const selected = videos.find(v => v.type === e.target.value);
                if (selected) {
                    if (selected.type === '3d_ply') {
                        this.load3DViewer(selected);
                    } else {
                        this.loadVideo(selected);
                    }
                }
            });
        });
        
        // Load first video
        if (videos.length > 0) {
            if (videos[0].type === '3d_ply') {
                this.load3DViewer(videos[0]);
            } else {
                this.loadVideo(videos[0]);
            }
        }
        
        // Update info
        this.analysisInfo.innerHTML = `
            <div class="flex justify-between">
                <span class="text-gray-600">Configuration</span>
                <span class="font-medium">${this.getConfigDisplayName(this.currentAnalysisExp.cfg)}</span>
            </div>
            <div class="flex justify-between">
                <span class="text-gray-600">Seed</span>
                <span class="font-medium">${this.currentAnalysisExp.seed}</span>
            </div>
            <div class="flex justify-between">
                <span class="text-gray-600">Total Frames</span>
                <span class="font-medium">${this.currentAnalysisExp.metadata.num_frames || 'N/A'}</span>
            </div>
            <div class="flex justify-between">
                <span class="text-gray-600">Resolution</span>
                <span class="font-medium">${this.currentAnalysisExp.metadata.resolution ? this.currentAnalysisExp.metadata.resolution.join('×') : 'N/A'}</span>
            </div>
            <div class="flex justify-between">
                <span class="text-gray-600">Voxel Resolution</span>
                <span class="font-medium">${this.currentAnalysisExp.metadata.voxel_resolution || 'N/A'}</span>
            </div>
        `;
    }
    
    loadVideo(video) {
        this.currentVideoType = video.type;
        
        // Ensure we have a video element
        if (!this.videoWrapper.querySelector('#analysisVideo')) {
            this.videoWrapper.innerHTML = `
                <video id="analysisVideo" muted preload="metadata">
                    Your browser does not support the video tag.
                </video>
            `;
            this.analysisVideo = document.getElementById('analysisVideo');
            this.attachVideoEventListeners();
        }
        
        // Clear any existing source
        this.analysisVideo.pause();
        this.stopProgressAnimation();
        this.analysisVideo.src = '';
        
        // Set new source
        const videoUrl = `/api/media/${video.path}`;
        
        // Test if file exists first
        fetch(`/api/test/file/${video.path}`)
            .then(res => res.json())
            .then(data => {
                if (!data.exists) {
                    console.error(`Video file not found: ${video.path}`);
                    this.showVideoError(`Video file not found: ${video.title}`);
                } else if (data.size === 0) {
                    console.error(`Video file is empty: ${video.path}`);
                    this.showVideoError(`Video file is empty: ${video.title}`);
                } else {
                    // File exists and has content
                    console.log(`Loading video: ${video.path} (${data.size} bytes)`);
                    this.analysisVideo.src = videoUrl;
                    this.analysisVideo.load();
                }
            })
            .catch(err => {
                console.error('Error checking file:', err);
                // Try loading anyway
                this.analysisVideo.src = videoUrl;
                this.analysisVideo.load();
            });
        
        // Reset controls
        this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
        this.videoSeeker.value = 0;
        if (this.videoProgressFill) {
            this.videoProgressFill.style.width = '0%';
        }
        
        // Apply loop setting
        this.analysisVideo.loop = this.loopVideo.checked;
    }
    
    showVideoError(message) {
        // Show error message in the video container
        this.videoWrapper.innerHTML = `
            <div class="flex items-center justify-center w-full h-full">
                <div class="text-center text-white p-8">
                    <i class="fas fa-exclamation-triangle text-4xl mb-4 text-yellow-500"></i>
                    <p class="text-lg">${message}</p>
                    <p class="text-sm mt-2 text-gray-400">Please check if the file exists in the experiments folder</p>
                </div>
            </div>
        `;
        
        // Re-create video element for future use
        const newVideo = document.createElement('video');
        newVideo.id = 'analysisVideo';
        newVideo.muted = true;
        newVideo.preload = 'metadata';
        newVideo.innerHTML = 'Your browser does not support the video tag.';
        
        // Store reference and re-attach event listeners
        this.analysisVideo = newVideo;
        this.attachVideoEventListeners();
    }
    
    attachVideoEventListeners() {
        this.analysisVideo.addEventListener('timeupdate', () => {
            this.updateVideoProgress();
        });
        
        this.analysisVideo.addEventListener('progress', () => {
            this.updateVideoProgress();
        });
        
        this.analysisVideo.addEventListener('loadedmetadata', () => {
            this.updateVideoProgress();
            // Delay resize to ensure container dimensions are available
            setTimeout(() => {
                this.resizeVideo();
            }, 100);
            const totalFrames = Math.floor(this.analysisVideo.duration * this.frameRate);
            this.totalFrames.textContent = totalFrames;
        });
        
        this.analysisVideo.addEventListener('error', (e) => {
            console.error('Video error:', e);
            const video = e.target;
            if (video.error) {
                console.error('Video error code:', video.error.code);
                console.error('Video error message:', video.error.message);
                
                // Handle codec errors specifically
                if (video.error.message && video.error.message.includes('DEMUXER_ERROR')) {
                    console.warn('Video codec not supported by browser, server should transcode');
                }
            }
        });
        
        this.analysisVideo.addEventListener('ended', () => {
            this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
            this.stopProgressAnimation();
        });
    }
    
    closeInteractiveAnalysis() {
        this.interactiveAnalysis.classList.add('hidden');
        this.analysisVideo.pause();
        this.analysisVideo.src = '';
        
        // Clean up 3D viewer if exists
        if (this.threeViewer) {
            this.threeViewer.dispose();
            this.threeViewer = null;
        }
        
        // Reset containers to default state
        this.videoContainer.classList.remove('hidden');
        this.videoControls.classList.remove('hidden');
        this.threeDContainer.classList.add('hidden');
    }
    
    togglePlayPause() {
        if (this.analysisVideo.paused) {
            this.analysisVideo.play();
            this.playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
            this.startProgressAnimation();
        } else {
            this.analysisVideo.pause();
            this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
            this.stopProgressAnimation();
        }
    }
    
    startProgressAnimation() {
        const animate = () => {
            if (!this.analysisVideo.paused) {
                this.updateVideoProgress();
                this.animationFrame = requestAnimationFrame(animate);
            }
        };
        this.animationFrame = requestAnimationFrame(animate);
    }
    
    stopProgressAnimation() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }
    
    stepFrame(direction) {
        const frameTime = 1 / this.frameRate;
        this.analysisVideo.currentTime += frameTime * direction;
        if (this.analysisVideo.paused) {
            this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
        }
    }
    
    toggleFullscreen() {
        const container = this.interactiveAnalysis;
        if (!document.fullscreenElement) {
            container.requestFullscreen();
            this.fullscreenBtn.innerHTML = '<i class="fas fa-compress"></i>';
        } else {
            document.exitFullscreen();
            this.fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';
        }
    }
    
    resizeVideo() {
        // Check if video element still exists
        if (!this.analysisVideo || !this.analysisVideo.videoWidth || !this.analysisVideo.videoHeight) {
            return;
        }
        
        // Get video natural dimensions
        const videoWidth = this.analysisVideo.videoWidth;
        const videoHeight = this.analysisVideo.videoHeight;
        const aspectRatio = videoWidth / videoHeight;
        
        // Get container dimensions
        const container = this.videoWrapper.parentElement;
        const sidebar = document.querySelector('.w-80.bg-white.border-l');
        const sidebarWidth = sidebar ? sidebar.offsetWidth : 320;
        
        // Get the controls container height
        const controlsContainer = document.getElementById('videoControls');
        const controlsHeight = controlsContainer ? controlsContainer.offsetHeight : 120;
        
        // Calculate available space more accurately
        const headerHeight = 88; // Header with title and close button
        const padding = 32; // Total vertical padding
        const availableWidth = window.innerWidth - sidebarWidth - 80; // Account for sidebar and padding
        const availableHeight = window.innerHeight - headerHeight - controlsHeight - padding;
        
        let width, height;
        
        // Calculate dimensions to fit available space while maintaining aspect ratio
        if (availableWidth / availableHeight > aspectRatio) {
            // Height is the limiting factor
            height = availableHeight;
            width = height * aspectRatio;
        } else {
            // Width is the limiting factor
            width = availableWidth;
            height = width / aspectRatio;
        }
        
        // Apply dimensions to wrapper to fit video exactly
        this.videoWrapper.style.width = `${Math.floor(width)}px`;
        this.videoWrapper.style.height = `${Math.floor(height)}px`;
        
        // Ensure video fills the wrapper exactly
        this.analysisVideo.style.width = '100%';
        this.analysisVideo.style.height = '100%';
        this.analysisVideo.style.objectFit = 'fill';
    }
    
    updateVideoProgress() {
        if (this.analysisVideo.duration && !isNaN(this.analysisVideo.duration)) {
            const progress = (this.analysisVideo.currentTime / this.analysisVideo.duration) * 100;
            
            // Update slider value
            if (!this.isDragging) {
                this.videoSeeker.value = progress;
            }
            
            // Always update progress fill to match current time
            if (this.videoProgressFill) {
                this.videoProgressFill.style.width = `${progress}%`;
            }
            
            // Update time display
            this.currentTime.textContent = this.formatTime(this.analysisVideo.currentTime);
            this.duration.textContent = this.formatTime(this.analysisVideo.duration);
            
            // Update frame counter
            const currentFrame = Math.floor(this.analysisVideo.currentTime * this.frameRate);
            this.currentFrame.textContent = currentFrame;
        }
    }
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    getConfigDisplayName(cfg) {
        // Map internal configuration names to display names
        const nameMap = {
            'Cats': 'Swing',
            'Cat': 'Swing',
            'Head': 'Pano',
            'Heads': 'Pano'
        };
        return nameMap[cfg] || cfg;
    }
    
    getConfigIcon(cfg) {
        // Map configuration names to appropriate icons
        if (cfg === 'Cats' || cfg === 'Cat') {
            return 'fa-sync-alt'; // Swing motion icon
        } else if (cfg === 'Head' || cfg === 'Heads') {
            return 'fa-panorama'; // Panoramic icon for pano rotation
        } else {
            return 'fa-image'; // Default icon
        }
    }
    
    // 3D Viewer Methods
    load3DViewer(config) {
        // Hide video container, show 3D container
        this.videoContainer.classList.add('hidden');
        this.videoControls.classList.add('hidden');
        this.threeDContainer.classList.remove('hidden');
        
        // Clean up existing viewer
        if (this.threeViewer) {
            this.threeViewer.dispose();
            this.threeViewer = null;
        }
        
        // Populate PLY selector
        this.plySelector.innerHTML = config.meshFiles.map(file => {
            // Extract metric name from filename (e.g., mesh_lpips.ply -> LPIPS)
            const match = file.match(/mesh_(.+)\.ply$/);
            const metricName = match ? this.formatMetricName(match[1]) : file;
            return `<option value="${file}">${metricName}</option>`;
        }).join('');
        
        // Initialize Three.js viewer
        this.initThreeViewer();
        
        // Initialize colorbar
        this.drawColorbar();
        
        // Load first PLY file
        if (config.meshFiles.length > 0) {
            this.loadPLYModel(config.meshFiles[0]);
            this.updateMetricInfo(config.meshFiles[0]);
            this.updateColorbarLabels(config.meshFiles[0].match(/mesh_(.+)\.ply$/)?.[1] || '');
            // Show metric info by default
            setTimeout(() => this.showMetricInfo(), 100);
        }
    }
    
    initThreeViewer() {
        // Wait for ThreeViewer to be available
        if (!window.ThreeViewer) {
            setTimeout(() => this.initThreeViewer(), 100);
            return;
        }
        
        this.threeViewer = new window.ThreeViewer(this.threeDViewer);
    }
    
    loadPLYModel(filepath) {
        if (!this.threeViewer) return;
        
        // Extract and set current metric
        const match = filepath.match(/mesh_(.+)\.ply$/);
        if (match) {
            this.threeViewer.setCurrentMetric(match[1]);
        }
        
        // Show loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75 z-10';
        loadingDiv.innerHTML = `
            <div class="text-center">
                <div class="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
                <p class="mt-4 text-white">Loading 3D model...</p>
            </div>
        `;
        this.threeDViewer.appendChild(loadingDiv);
        
        // Load PLY file
        this.threeViewer.loadPLY(
            `/api/media/${filepath}`,
            (xhr) => {
                // Progress callback
                if (xhr.lengthComputable) {
                    const percentComplete = xhr.loaded / xhr.total * 100;
                    console.log(`Loading: ${Math.round(percentComplete)}%`);
                }
            },
            (error) => {
                console.error('Error loading PLY file:', error);
                loadingDiv.innerHTML = `
                    <div class="text-center">
                        <i class="fas fa-exclamation-triangle text-4xl text-red-500 mb-4"></i>
                        <p class="text-white">Failed to load 3D model</p>
                    </div>
                `;
            },
            () => {
                // On successful load, remove loading indicator
                if (loadingDiv.parentNode) {
                    loadingDiv.remove();
                }
            }
        );
    }
    
    resizeThreeViewer() {
        if (!this.threeViewer) return;
        
        const width = this.threeDViewer.clientWidth;
        const height = this.threeDViewer.clientHeight;
        
        this.threeViewer.resize(width, height);
    }
    
    loadVideo(video) {
        // Show video container, hide 3D container
        this.videoContainer.classList.remove('hidden');
        this.videoControls.classList.remove('hidden');
        this.threeDContainer.classList.add('hidden');
        
        // Clean up 3D viewer if exists
        if (this.threeViewer) {
            this.threeViewer.dispose();
            this.threeViewer = null;
        }
        
        this.currentVideoType = video.type;
        
        // Rest of the original loadVideo code...
        // Ensure we have a video element
        if (!this.videoWrapper.querySelector('#analysisVideo')) {
            this.videoWrapper.innerHTML = `
                <video id="analysisVideo" muted preload="metadata">
                    Your browser does not support the video tag.
                </video>
            `;
            this.analysisVideo = document.getElementById('analysisVideo');
            this.attachVideoEventListeners();
        }
        
        // Clear any existing source
        this.analysisVideo.pause();
        this.stopProgressAnimation();
        this.analysisVideo.src = '';
        
        // Set new source
        const videoUrl = `/api/media/${video.path}`;
        
        // Test if file exists first
        fetch(`/api/test/file/${video.path}`)
            .then(res => res.json())
            .then(data => {
                if (!data.exists) {
                    console.error(`Video file not found: ${video.path}`);
                    this.showVideoError(`Video file not found: ${video.title}`);
                } else if (data.size === 0) {
                    console.error(`Video file is empty: ${video.path}`);
                    this.showVideoError(`Video file is empty: ${video.title}`);
                } else {
                    // File exists and has content
                    console.log(`Loading video: ${video.path} (${data.size} bytes)`);
                    this.analysisVideo.src = videoUrl;
                    this.analysisVideo.load();
                }
            })
            .catch(err => {
                console.error('Error checking file:', err);
                // Try loading anyway
                this.analysisVideo.src = videoUrl;
                this.analysisVideo.load();
            });
        
        // Reset controls
        this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
        this.videoSeeker.value = 0;
        if (this.videoProgressFill) {
            this.videoProgressFill.style.width = '0%';
        }
        
        // Apply loop setting
        this.analysisVideo.loop = this.loopVideo.checked;
    }
    
    updateMetricInfo(filepath) {
        // Extract metric name from filename
        const match = filepath.match(/mesh_(.+)\.ply$/);
        if (match) {
            this.currentMetric = match[1];
        }
    }
    
    showMetricInfo() {
        if (!this.currentMetric || !window.getMetricInfo) {
            return;
        }
        
        const info = window.getMetricInfo(this.currentMetric);
        
        // Update panel content
        this.metricName.textContent = info.name;
        this.metricCategory.textContent = info.category;
        this.metricDescription.textContent = info.description;
        
        // Build interpretation HTML with magma colormap colors
        let interpretationHTML = '';
        
        // Determine if metric is reversed (higher is better)
        const higherIsBetter = this.currentMetric && (
            this.currentMetric.includes('cosine_dist') || 
            this.currentMetric.includes('l2') || 
            this.currentMetric.includes('l1') ||
            this.currentMetric.includes('psnr') ||
            this.currentMetric.includes('temporal_identity_consistency') ||
            this.currentMetric.includes('temporal_visual_smoothness') ||
            this.currentMetric.includes('anonymization_coverage')
        );
        
        if (info.interpretation.good) {
            const goodColor = higherIsBetter ? '#fcffa4' : '#000004'; // Hot or cold based on metric
            const goodTextColor = higherIsBetter ? '#796700' : '#4a4a4a';
            interpretationHTML += `<div class="flex items-center space-x-2">
                <span class="w-3 h-3 rounded-full" style="background-color: ${goodColor}; border: 1px solid #e5e7eb;"></span>
                <span style="color: ${goodTextColor}">Good: ${info.interpretation.good}</span>
            </div>`;
        }
        if (info.interpretation.medium) {
            interpretationHTML += `<div class="flex items-center space-x-2">
                <span class="w-3 h-3 rounded-full" style="background-color: #cd4071; border: 1px solid #e5e7eb;"></span>
                <span style="color: #7a2650">Medium: ${info.interpretation.medium}</span>
            </div>`;
        }
        if (info.interpretation.poor) {
            const poorColor = higherIsBetter ? '#000004' : '#fcffa4'; // Cold or hot based on metric
            const poorTextColor = higherIsBetter ? '#4a4a4a' : '#796700';
            interpretationHTML += `<div class="flex items-center space-x-2">
                <span class="w-3 h-3 rounded-full" style="background-color: ${poorColor}; border: 1px solid #e5e7eb;"></span>
                <span style="color: ${poorTextColor}">Poor: ${info.interpretation.poor}</span>
            </div>`;
        }
        this.metricInterpretation.innerHTML = interpretationHTML;
        
        // Add details and formula
        let detailsHTML = info.details || '';
        if (info.formula) {
            detailsHTML += `<div class="mt-2 pt-2 border-t border-gray-200">
                <span class="font-semibold">Formula:</span> <code class="text-xs bg-gray-100 px-1 py-0.5 rounded">${info.formula}</code>
            </div>`;
        }
        this.metricDetails.innerHTML = detailsHTML;
        
        // Show panel
        this.metricInfoPanel.classList.remove('hidden');
    }
    
    drawColorbar() {
        const canvas = this.colorbarCanvas;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Create magma colormap gradient
        const gradient = ctx.createLinearGradient(0, height, 0, 0);
        
        // Magma colormap approximation (from dark purple to bright yellow)
        gradient.addColorStop(0.0, '#000004');
        gradient.addColorStop(0.1, '#180f3e');
        gradient.addColorStop(0.2, '#451077');
        gradient.addColorStop(0.3, '#721f81');
        gradient.addColorStop(0.4, '#9e2f7f');
        gradient.addColorStop(0.5, '#cd4071');
        gradient.addColorStop(0.6, '#f1605d');
        gradient.addColorStop(0.7, '#fd9668');
        gradient.addColorStop(0.8, '#feca8d');
        gradient.addColorStop(0.9, '#fcfdbf');
        gradient.addColorStop(1.0, '#fcffa4');
        
        // Fill gradient
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        
        // Add border
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 1;
        ctx.strokeRect(0, 0, width, height);
    }
    
    updateColorbarLabels(metric) {
        // Get metric info to determine appropriate scale
        if (!window.getMetricInfo) return;
        
        const info = window.getMetricInfo(metric);
        
        // Update labels and meanings based on metric type
        if (info.category === 'Identity Distance') {
            if (metric.includes('cosine_sim')) {
                this.colorbarMax.textContent = '1.0';
                this.colorbarMid.textContent = '0.5';
                this.colorbarMin.textContent = '0.0';
                this.colorbarHotMeaning.textContent = 'Similar';
                this.colorbarColdMeaning.textContent = 'Different';
            } else if (metric.includes('cosine_dist')) {
                this.colorbarMax.textContent = '1.0';
                this.colorbarMid.textContent = '0.5';
                this.colorbarMin.textContent = '0.0';
                this.colorbarHotMeaning.textContent = 'Different';
                this.colorbarColdMeaning.textContent = 'Similar';
            } else if (metric.includes('l2')) {
                this.colorbarMax.textContent = '30';
                this.colorbarMid.textContent = '15';
                this.colorbarMin.textContent = '0';
                this.colorbarHotMeaning.textContent = 'Far Apart';
                this.colorbarColdMeaning.textContent = 'Close';
            }
        } else if (info.category === 'Security Threshold') {
            this.colorbarMax.textContent = '1.0';
            this.colorbarMid.textContent = '0.5';
            this.colorbarMin.textContent = '0.0';
            this.colorbarHotMeaning.textContent = 'Matched';
            this.colorbarColdMeaning.textContent = 'Not Matched';
        } else if (info.category === 'Perceptual Quality') {
            this.colorbarMax.textContent = '1.0';
            this.colorbarMid.textContent = '0.5';
            this.colorbarMin.textContent = '0.0';
            this.colorbarHotMeaning.textContent = 'Different';
            this.colorbarColdMeaning.textContent = 'Similar';
        } else if (info.category === 'Image Quality') {
            if (metric.includes('psnr')) {
                this.colorbarMax.textContent = '50 dB';
                this.colorbarMid.textContent = '35 dB';
                this.colorbarMin.textContent = '20 dB';
                this.colorbarHotMeaning.textContent = 'High Quality';
                this.colorbarColdMeaning.textContent = 'Low Quality';
            } else if (metric.includes('ssim')) {
                this.colorbarMax.textContent = '1.0';
                this.colorbarMid.textContent = '0.5';
                this.colorbarMin.textContent = '0.0';
                this.colorbarHotMeaning.textContent = 'Preserved';
                this.colorbarColdMeaning.textContent = 'Degraded';
            }
        } else if (metric.includes('temporal')) {
            this.colorbarMax.textContent = '1.0';
            this.colorbarMid.textContent = '0.5';
            this.colorbarMin.textContent = '0.0';
            this.colorbarHotMeaning.textContent = 'Stable';
            this.colorbarColdMeaning.textContent = 'Unstable';
        } else {
            // Default
            this.colorbarMax.textContent = '1.0';
            this.colorbarMid.textContent = '0.5';
            this.colorbarMin.textContent = '0.0';
            this.colorbarHotMeaning.textContent = 'High';
            this.colorbarColdMeaning.textContent = 'Low';
        }
        
        // Draw the colorbar
        this.drawColorbar();
    }
    
    formatMetricName(metric) {
        // Handle special cases first
        const specialCases = {
            'lpips': 'LPIPS',
            'psnr': 'PSNR',
            'ssim': 'SSIM',
            'fid': 'FID',
            'l2': 'L2 Distance',
            'cosine_sim': 'Cosine Similarity',
            'cosine_dist': 'Cosine Distance',
            'far_1e3': 'FAR@1e-3',
            'far_1e4': 'FAR@1e-4',
            'far_1e5': 'FAR@1e-5'
        };
        
        if (specialCases[metric.toLowerCase()]) {
            return specialCases[metric.toLowerCase()];
        }
        
        // For other metrics, replace underscores with spaces and capitalize each word
        return metric
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
            .join(' ');
    }
    
    setupVisibilityObserver() {
        // Create an observer to watch for visibility changes
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    const target = mutation.target;
                    if (target === this.interactiveAnalysis) {
                        // Check if interactive analysis is now visible
                        if (!target.classList.contains('hidden')) {
                            // Delay to ensure DOM is ready
                            setTimeout(() => {
                                if (!this.threeDContainer.classList.contains('hidden')) {
                                    this.resizeThreeViewer();
                                } else if (this.analysisVideo && this.analysisVideo.videoWidth) {
                                    this.resizeVideo();
                                }
                            }, 100);
                        }
                    }
                }
            });
        });
        
        // Start observing the interactive analysis element
        if (this.interactiveAnalysis) {
            observer.observe(this.interactiveAnalysis, { 
                attributes: true, 
                attributeFilter: ['class'] 
            });
        }
    }
}

// Initialize application
const app = new CompactExperimentsViewer();
// Make app globally available for onclick handlers
window.app = app;