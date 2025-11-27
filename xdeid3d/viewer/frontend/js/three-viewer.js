import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class ThreeViewer {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.currentMesh = null;
        this.animationId = null;
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.hoveredPoint = null;
        this.metricValues = null;
        this.colorScale = null;
        
        this.init();
    }
    
    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 0, 5);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(5, 5, 5);
        this.scene.add(directionalLight);
        
        // Add controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 1;
        this.controls.maxDistance = 20;
        
        // Add mouse event listeners
        this.renderer.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.renderer.domElement.addEventListener('click', (e) => this.onMouseClick(e));
        
        // Create hover tooltip
        this.createTooltip();
        
        // Start animation loop
        this.animate();
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    loadPLY(url, onProgress, onError, onLoad) {
        // Remove existing mesh
        if (this.currentMesh) {
            this.scene.remove(this.currentMesh);
            this.currentMesh.geometry.dispose();
            this.currentMesh.material.dispose();
            this.currentMesh = null;
        }
        
        const loader = new PLYLoader();
        loader.load(
            url,
            (geometry) => {
                // Center the geometry
                geometry.computeBoundingBox();
                const center = geometry.boundingBox.getCenter(new THREE.Vector3());
                geometry.translate(-center.x, -center.y, -center.z);
                
                // Create material with vertex colors
                const material = new THREE.MeshPhongMaterial({
                    vertexColors: true,
                    side: THREE.DoubleSide,
                    shininess: 50,
                    specular: 0x222222
                });
                
                // Create mesh
                const mesh = new THREE.Mesh(geometry, material);
                this.scene.add(mesh);
                this.currentMesh = mesh;
                
                // Adjust camera to fit the model
                const box = new THREE.Box3().setFromObject(mesh);
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = this.camera.fov * (Math.PI / 180);
                let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
                cameraZ *= 1.5; // Add some padding
                
                this.camera.position.set(cameraZ, cameraZ/2, cameraZ);
                this.camera.lookAt(0, 0, 0);
                this.controls.target.set(0, 0, 0);
                this.controls.update();
                
                if (onLoad) onLoad();
            },
            onProgress,
            onError
        );
    }
    
    resize(width, height) {
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.currentMesh) {
            this.scene.remove(this.currentMesh);
            this.currentMesh.geometry.dispose();
            this.currentMesh.material.dispose();
        }
        if (this.tooltip) {
            this.tooltip.remove();
        }
        this.renderer.dispose();
        this.renderer.domElement.remove();
    }
    
    createTooltip() {
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'absolute bg-gray-900 text-white px-3 py-2 rounded-lg text-xs pointer-events-none z-50 hidden shadow-xl max-w-xs';
        this.container.appendChild(this.tooltip);
    }
    
    setCurrentMetric(metricName) {
        this.currentMetric = metricName;
    }
    
    onMouseMove(event) {
        if (!this.currentMesh) return;
        
        // Calculate mouse position in normalized device coordinates
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        // Update raycaster
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        // Check for intersections
        const intersects = this.raycaster.intersectObject(this.currentMesh);
        
        if (intersects.length > 0) {
            const intersection = intersects[0];
            const face = intersection.face;
            const geometry = this.currentMesh.geometry;
            
            // Get vertex colors for the face
            const colors = geometry.attributes.color;
            if (colors) {
                // Average the colors of the three vertices of the face
                const color1 = new THREE.Color(
                    colors.getX(face.a),
                    colors.getY(face.a),
                    colors.getZ(face.a)
                );
                const color2 = new THREE.Color(
                    colors.getX(face.b),
                    colors.getY(face.b),
                    colors.getZ(face.b)
                );
                const color3 = new THREE.Color(
                    colors.getX(face.c),
                    colors.getY(face.c),
                    colors.getZ(face.c)
                );
                
                // Calculate average color
                const avgColor = new THREE.Color();
                avgColor.r = (color1.r + color2.r + color3.r) / 3;
                avgColor.g = (color1.g + color2.g + color3.g) / 3;
                avgColor.b = (color1.b + color2.b + color3.b) / 3;
                
                // Estimate metric value from color (reverse of magma colormap)
                const metricValue = this.colorToMetricValue(avgColor);
                
                // Get metric interpretation
                const interpretation = this.getMetricInterpretation(metricValue);
                
                // Update tooltip with detailed information
                const formattedMetric = this.currentMetric ? this.formatMetricName(this.currentMetric) : 'Metric';
                const metricInfo = this.getDetailedMetricInfo();
                
                // Create a more informative tooltip
                let tooltipContent = `<div class="font-semibold mb-1">${formattedMetric}: ${metricValue.toFixed(3)}</div>`;
                
                // Add the evaluation status with color
                if (interpretation.statusText) {
                    tooltipContent += `<div class="text-sm font-medium ${interpretation.statusClass} mb-1">${interpretation.statusText}</div>`;
                }
                
                // Add contextual description
                if (interpretation.description) {
                    tooltipContent += `<div class="text-xs opacity-90 mb-1">${interpretation.description}</div>`;
                }
                
                // Add the range info if available
                if (metricInfo.range) {
                    tooltipContent += `<div class="text-xs opacity-70">${metricInfo.range}</div>`;
                }
                
                this.tooltip.innerHTML = tooltipContent;
                
                // Position tooltip
                const tooltipX = event.clientX - rect.left + 10;
                const tooltipY = event.clientY - rect.top - 10;
                
                // Keep tooltip within bounds
                const maxX = rect.width - 200; // Approximate tooltip width
                const maxY = rect.height - 100; // Approximate tooltip height
                
                this.tooltip.style.left = `${Math.min(tooltipX, maxX)}px`;
                this.tooltip.style.top = `${Math.max(10, Math.min(tooltipY, maxY))}px`;
                this.tooltip.classList.remove('hidden');
                
                // Change cursor
                this.renderer.domElement.style.cursor = 'pointer';
            }
        } else {
            // Hide tooltip
            this.tooltip.classList.add('hidden');
            this.renderer.domElement.style.cursor = 'auto';
        }
    }
    
    onMouseClick(event) {
        // Could be used for more detailed popup on click
    }
    
    colorToMetricValue(color) {
        // This is an approximation - we're reversing the magma colormap
        // Magma goes from dark purple/black (0) to bright yellow/white (1)
        // This is a simplified estimation based on brightness
        const brightness = (color.r + color.g + color.b) / 3;
        
        // Apply non-linear mapping to better match magma colormap
        let value = brightness;
        if (brightness < 0.3) {
            value = brightness * 0.5;
        } else if (brightness < 0.7) {
            value = 0.15 + (brightness - 0.3) * 1.125;
        } else {
            value = 0.6 + (brightness - 0.7) * 1.33;
        }
        
        return Math.min(1, Math.max(0, value));
    }
    
    updateColorScale(minValue, maxValue) {
        this.colorScale = { min: minValue, max: maxValue };
    }
    
    getMetricInterpretation(value) {
        if (!this.currentMetric || !window.getMetricInfo) {
            return { statusText: 'Unknown', statusClass: '', description: '' };
        }
        
        const info = window.getMetricInfo(this.currentMetric);
        let statusText = '';
        let statusClass = '';
        let description = '';
        
        // Use interpretation from metric info if available
        if (info.interpretation) {
            // Check value against interpretation ranges
            const { good, medium, poor } = info.interpretation;
            
            // Parse the interpretation strings to determine status
            if (this.matchesInterpretation(value, good)) {
                statusText = this.extractStatusText(good, info.category);
                statusClass = 'text-green-400';
                description = this.getDescriptionForStatus('good', info);
            } else if (medium && this.matchesInterpretation(value, medium)) {
                statusText = this.extractStatusText(medium, info.category);
                statusClass = 'text-yellow-400';
                description = this.getDescriptionForStatus('medium', info);
            } else if (this.matchesInterpretation(value, poor)) {
                statusText = this.extractStatusText(poor, info.category);
                statusClass = 'text-red-400';
                description = this.getDescriptionForStatus('poor', info);
            } else {
                // If no range matches, provide a contextual default
                if (info.category === 'Identity Distance') {
                    // For identity metrics, determine based on common thresholds
                    if (this.currentMetric.includes('cosine_sim')) {
                        if (value < 0.3) {
                            statusText = '✓ Excellent';
                            statusClass = 'text-green-400';
                            description = 'Strong anonymization';
                        } else if (value > 0.7) {
                            statusText = '✗ Very Poor';
                            statusClass = 'text-red-400';
                            description = 'Weak anonymization';
                        } else {
                            statusText = '⚠ Moderate';
                            statusClass = 'text-yellow-400';
                            description = 'Partial anonymization';
                        }
                    } else if (this.currentMetric.includes('cosine_dist')) {
                        if (value > 0.7) {
                            statusText = '✓ Excellent';
                            statusClass = 'text-green-400';
                            description = 'Strong anonymization';
                        } else if (value < 0.3) {
                            statusText = '✗ Very Poor';
                            statusClass = 'text-red-400';
                            description = 'Weak anonymization';
                        } else {
                            statusText = '⚠ Moderate';
                            statusClass = 'text-yellow-400';
                            description = 'Partial anonymization';
                        }
                    }
                } else {
                    // Generic fallback
                    statusText = `Value: ${value.toFixed(3)}`;
                    statusClass = 'text-gray-400';
                    description = '';
                }
            }
            
            return { statusText, statusClass, description };
        }
        
        // Fallback - provide specific evaluations based on metric type
        return this.getSpecificMetricEvaluation(value, info);
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
    
    getDetailedMetricInfo() {
        if (!this.currentMetric || !window.getMetricInfo) {
            return { range: '', optimal: '' };
        }
        
        const info = window.getMetricInfo(this.currentMetric);
        
        // Extract range and optimal from the metric info
        let range = '';
        let optimal = '';
        
        if (info.details) {
            // Extract range information from details
            const rangeMatch = info.details.match(/Range: ([^.]+)/i);
            if (rangeMatch) {
                range = rangeMatch[0];
            }
        }
        
        // Build optimal string from interpretation
        if (info.interpretation && info.interpretation.good) {
            const goodDesc = info.interpretation.good;
            if (info.category === 'Identity Distance') {
                if (goodDesc.includes('<')) {
                    optimal = `Optimal: ${goodDesc.split('(')[0].trim()} for good anonymization`;
                } else if (goodDesc.includes('>')) {
                    optimal = `Optimal: ${goodDesc.split('(')[0].trim()} for good anonymization`;
                }
            } else if (info.category === 'Security Threshold') {
                optimal = `Optimal: ${goodDesc}`;
            } else if (info.category === 'Perceptual Quality' || info.category === 'Image Quality') {
                optimal = `Optimal: ${goodDesc.split('(')[0].trim()}`;
            } else if (info.category === 'Temporal Consistency') {
                optimal = `Optimal: ${goodDesc.split('(')[0].trim()}`;
            } else if (info.category === 'Attribute Preservation') {
                optimal = `Optimal: ${goodDesc.split('(')[0].trim()}`;
            }
        }
        
        // Fallback to formula if available
        if (!range && info.formula) {
            range = `Formula: ${info.formula}`;
        }
        
        return { range, optimal };
    }
    
    matchesInterpretation(value, interpretationStr) {
        if (!interpretationStr) return false;
        
        // Handle special cases for metrics that need scaling
        let scaledValue = value;
        if (this.currentMetric && this.currentMetric.includes('psnr')) {
            // PSNR is typically in dB, scale from normalized 0-1 to 20-50 dB range
            scaledValue = value * 30 + 20;
        } else if (this.currentMetric && this.currentMetric.includes('l2') && this.currentMetric.includes('arcface')) {
            // L2 distance for ArcFace, scale from 0-1 to 0-30 range
            scaledValue = value * 30;
        }
        
        // Extract numeric comparisons from interpretation strings
        const ltMatch = interpretationStr.match(/< ([0-9.]+)/);  
        const gtMatch = interpretationStr.match(/> ([0-9.]+)/);
        const rangeMatch = interpretationStr.match(/([0-9.]+)\s*-\s*([0-9.]+)/);
        const eqMatch = interpretationStr.match(/= ([0-9.]+)/);
        const percentMatch = interpretationStr.match(/> ([0-9.]+)%/);
        
        if (percentMatch) {
            // Handle percentage comparisons
            return (scaledValue * 100) > parseFloat(percentMatch[1]);
        } else if (ltMatch) {
            return scaledValue < parseFloat(ltMatch[1]);
        } else if (gtMatch) {
            return scaledValue > parseFloat(gtMatch[1]);
        } else if (rangeMatch) {
            const min = parseFloat(rangeMatch[1]);
            const max = parseFloat(rangeMatch[2]);
            return scaledValue >= min && scaledValue <= max;
        } else if (eqMatch) {
            return Math.abs(scaledValue - parseFloat(eqMatch[1])) < 0.01;
        }
        
        // For security thresholds (FAR), check if it mentions "Not matched"
        if (interpretationStr.toLowerCase().includes('not matched')) {
            // Check against the threshold value mentioned in the string
            const thresholdMatch = interpretationStr.match(/similarity < ([0-9.]+)/);
            if (thresholdMatch) {
                return value < parseFloat(thresholdMatch[1]);
            }
        } else if (interpretationStr.toLowerCase().includes('matched')) {
            const thresholdMatch = interpretationStr.match(/similarity ≥ ([0-9.]+)/);
            if (thresholdMatch) {
                return value >= parseFloat(thresholdMatch[1]);
            }
        }
        
        return false;
    }
    
    extractStatusText(interpretationStr, category) {
        // For security thresholds, provide clear pass/fail
        if (category === 'Security Threshold') {
            if (interpretationStr.toLowerCase().includes('not matched')) {
                return '✓ Secure (Not Matched)';
            } else if (interpretationStr.toLowerCase().includes('matched')) {
                return '✗ Not Secure (Matched)';
            }
        }
        
        // For other metrics, extract the quality level
        if (interpretationStr.toLowerCase().includes('good')) {
            return '✓ Good';
        } else if (interpretationStr.toLowerCase().includes('medium') || interpretationStr.toLowerCase().includes('acceptable')) {
            return '⚠ Medium';
        } else if (interpretationStr.toLowerCase().includes('poor') || interpretationStr.toLowerCase().includes('low')) {
            return '✗ Poor';
        }
        
        // Try to extract from parentheses
        const parenMatch = interpretationStr.match(/\(([^)]+)\)/);
        if (parenMatch) {
            const extracted = parenMatch[1];
            // Add appropriate symbol based on content
            if (extracted.toLowerCase().includes('good') || extracted.toLowerCase().includes('stable') || extracted.toLowerCase().includes('preserved')) {
                return '✓ ' + extracted;
            } else if (extracted.toLowerCase().includes('poor') || extracted.toLowerCase().includes('unstable') || extracted.toLowerCase().includes('degraded')) {
                return '✗ ' + extracted;
            } else {
                return '⚠ ' + extracted;
            }
        }
        
        return interpretationStr;
    }
    
    getSpecificMetricEvaluation(value, info) {
        let statusText = '';
        let statusClass = '';
        let description = '';
        
        const metricLower = this.currentMetric ? this.currentMetric.toLowerCase() : '';
        
        // Identity Distance Metrics
        if (info.category === 'Identity Distance' || metricLower.includes('cosine') || metricLower.includes('l2') || metricLower.includes('l1')) {
            if (metricLower.includes('cosine_sim')) {
                if (value < 0.3) {
                    statusText = '✓ Excellent';
                    statusClass = 'text-green-400';
                    description = 'Strong de-identification';
                } else if (value < 0.5) {
                    statusText = '✓ Good';
                    statusClass = 'text-green-400';
                    description = 'Successful anonymization';
                } else if (value < 0.7) {
                    statusText = '⚠ Medium';
                    statusClass = 'text-yellow-400';
                    description = 'Partial anonymization';
                } else if (value < 0.7629) {
                    statusText = '✗ Poor';
                    statusClass = 'text-orange-400';
                    description = 'Weak anonymization';
                } else {
                    statusText = '✗ Failed';
                    statusClass = 'text-red-400';
                    description = 'Above FAR threshold';
                }
            } else if (metricLower.includes('cosine_dist')) {
                if (value > 0.7) {
                    statusText = '✓ Excellent';
                    statusClass = 'text-green-400';
                    description = 'Strong de-identification';
                } else if (value > 0.5) {
                    statusText = '✓ Good';
                    statusClass = 'text-green-400';
                    description = 'Successful anonymization';
                } else if (value > 0.3) {
                    statusText = '⚠ Medium';
                    statusClass = 'text-yellow-400';
                    description = 'Partial anonymization';
                } else if (value > 0.2371) {
                    statusText = '✗ Poor';
                    statusClass = 'text-orange-400';
                    description = 'Weak anonymization';
                } else {
                    statusText = '✗ Failed';
                    statusClass = 'text-red-400';
                    description = 'Identity preserved';
                }
            } else if (metricLower.includes('l2')) {
                const scaledValue = value * 30; // Scale to typical range
                if (scaledValue > 20) {
                    statusText = '✓ Excellent';
                    statusClass = 'text-green-400';
                    description = 'Strong de-identification';
                } else if (scaledValue > 15) {
                    statusText = '✓ Good';
                    statusClass = 'text-green-400';
                    description = 'Successful anonymization';
                } else if (scaledValue > 10) {
                    statusText = '⚠ Medium';
                    statusClass = 'text-yellow-400';
                    description = 'Partial anonymization';
                } else if (scaledValue > 5) {
                    statusText = '✗ Poor';
                    statusClass = 'text-orange-400';
                    description = 'Weak anonymization';
                } else {
                    statusText = '✗ Failed';
                    statusClass = 'text-red-400';
                    description = 'Identity preserved';
                }
            } else if (metricLower.includes('l1')) {
                if (value < 0.05) {
                    statusText = '✗ Weak';
                    statusClass = 'text-red-400';
                    description = 'Insufficient anonymization';
                } else if (value < 0.1) {
                    statusText = '⚠ Borderline';
                    statusClass = 'text-yellow-400';
                    description = 'Minimal anonymization';
                } else if (value < 0.2) {
                    statusText = '✓ Good';
                    statusClass = 'text-green-400';
                    description = 'Balanced anonymization';
                } else if (value < 0.3) {
                    statusText = '✓ Strong';
                    statusClass = 'text-green-400';
                    description = 'Good anonymization';
                } else if (value < 0.4) {
                    statusText = '⚠ Excessive';
                    statusClass = 'text-yellow-400';
                    description = 'Quality degrading';
                } else {
                    statusText = '✗ Poor Quality';
                    statusClass = 'text-red-400';
                    description = 'Too much distortion';
                }
            }
        }
        // Temporal Consistency Metrics
        else if (info.category === 'Temporal Consistency' || metricLower.includes('temporal')) {
            if (metricLower.includes('identity_consistency')) {
                if (value > 0.95) {
                    statusText = '✓ Excellent';
                    statusClass = 'text-green-400';
                    description = 'Very stable';
                } else if (value > 0.9) {
                    statusText = '✓ Good';
                    statusClass = 'text-green-400';
                    description = 'Stable across frames';
                } else if (value > 0.8) {
                    statusText = '⚠ Medium';
                    statusClass = 'text-yellow-400';
                    description = 'Some variation';
                } else if (value > 0.7) {
                    statusText = '✗ Poor';
                    statusClass = 'text-orange-400';
                    description = 'Noticeable flickering';
                } else {
                    statusText = '✗ Very Poor';
                    statusClass = 'text-red-400';
                    description = 'Severe instability';
                }
            } else if (metricLower.includes('visual_smoothness')) {
                if (value > 0.9) {
                    statusText = '✓ Very Smooth';
                    statusClass = 'text-green-400';
                    description = 'Excellent transitions';
                } else if (value > 0.8) {
                    statusText = '✓ Smooth';
                    statusClass = 'text-green-400';
                    description = 'Good transitions';
                } else if (value > 0.6) {
                    statusText = '⚠ Moderate';
                    statusClass = 'text-yellow-400';
                    description = 'Some jerkiness';
                } else if (value > 0.4) {
                    statusText = '✗ Jerky';
                    statusClass = 'text-orange-400';
                    description = 'Noticeable jumps';
                } else {
                    statusText = '✗ Very Jerky';
                    statusClass = 'text-red-400';
                    description = 'Severe discontinuity';
                }
            }
        }
        // Image Quality Metrics
        else if (info.category === 'Image Quality' || metricLower.includes('psnr') || metricLower.includes('ssim') || metricLower.includes('mse')) {
            if (metricLower.includes('psnr')) {
                const psnrValue = value * 30 + 20; // Scale to 20-50 dB range
                if (psnrValue > 45) {
                    statusText = '✗ Too High';
                    statusClass = 'text-orange-400';
                    description = `${psnrValue.toFixed(1)} dB - weak anonymization`;
                } else if (psnrValue > 35) {
                    statusText = '✓ Balanced';
                    statusClass = 'text-green-400';
                    description = `${psnrValue.toFixed(1)} dB - good`;
                } else if (psnrValue > 30) {
                    statusText = '✓ Good';
                    statusClass = 'text-green-400';
                    description = `${psnrValue.toFixed(1)} dB - strong anonymization`;
                } else if (psnrValue > 25) {
                    statusText = '⚠ Low';
                    statusClass = 'text-yellow-400';
                    description = `${psnrValue.toFixed(1)} dB - quality loss`;
                } else {
                    statusText = '✗ Poor';
                    statusClass = 'text-red-400';
                    description = `${psnrValue.toFixed(1)} dB - excessive distortion`;
                }
            } else if (metricLower.includes('ssim')) {
                if (value > 0.95) {
                    statusText = '✓ Excellent';
                    statusClass = 'text-green-400';
                    description = 'Nearly identical';
                } else if (value > 0.9) {
                    statusText = '✓ Good';
                    statusClass = 'text-green-400';
                    description = 'Well preserved';
                } else if (value > 0.8) {
                    statusText = '⚠ Medium';
                    statusClass = 'text-yellow-400';
                    description = 'Some degradation';
                } else if (value > 0.7) {
                    statusText = '✗ Poor';
                    statusClass = 'text-orange-400';
                    description = 'Notable changes';
                } else {
                    statusText = '✗ Very Poor';
                    statusClass = 'text-red-400';
                    description = 'Heavily degraded';
                }
            } else if (metricLower.includes('mse')) {
                if (value < 0.02) {
                    statusText = '✗ Weak';
                    statusClass = 'text-orange-400';
                    description = 'Insufficient anonymization';
                } else if (value < 0.05) {
                    statusText = '⚠ Borderline';
                    statusClass = 'text-yellow-400';
                    description = 'Minimal anonymization';
                } else if (value < 0.15) {
                    statusText = '✓ Balanced';
                    statusClass = 'text-green-400';
                    description = 'Good anonymization';
                } else if (value < 0.25) {
                    statusText = '⚠ High';
                    statusClass = 'text-yellow-400';
                    description = 'Quality degrading';
                } else {
                    statusText = '✗ Excessive';
                    statusClass = 'text-red-400';
                    description = 'Poor quality';
                }
            }
        }
        // Perceptual Quality (LPIPS - balance is key)
        else if (info.category === 'Perceptual Quality' || metricLower.includes('lpips')) {
            if (value < 0.2) {
                statusText = '✗ Too Similar';
                statusClass = 'text-orange-400';
                description = 'Weak anonymization';
            } else if (value < 0.3) {
                statusText = '⚠ Borderline';
                statusClass = 'text-yellow-400';
                description = 'Minimal changes';
            } else if (value < 0.7) {
                statusText = '✓ Balanced';
                statusClass = 'text-green-400';
                description = 'Good anonymization';
            } else if (value < 0.8) {
                statusText = '⚠ High';
                statusClass = 'text-yellow-400';
                description = 'Quality degrading';
            } else {
                statusText = '✗ Excessive';
                statusClass = 'text-red-400';
                description = 'Poor quality';
            }
        }
        // Default fallback
        else {
            if (value > 0.8) {
                statusText = 'High';
                statusClass = 'text-blue-400';
            } else if (value > 0.5) {
                statusText = 'Medium';
                statusClass = 'text-yellow-400';
            } else if (value > 0.2) {
                statusText = 'Low';
                statusClass = 'text-orange-400';
            } else {
                statusText = 'Very Low';
                statusClass = 'text-gray-400';
            }
            description = `${(value * 100).toFixed(1)}%`;
        }
        
        return { statusText, statusClass, description };
    }
    
    getDescriptionForStatus(status, info) {
        const descriptions = {
            'Identity Distance': {
                good: 'Identity successfully anonymized',
                medium: 'Partial identity protection',
                poor: 'Identity still recognizable'
            },
            'Temporal Consistency': {
                good: 'Stable across frames',
                medium: 'Some temporal variation',
                poor: 'Flickering/unstable'
            },
            'Image Quality': {
                good: 'High visual quality',
                medium: 'Acceptable quality',
                poor: 'Visible degradation'
            },
            'Perceptual Quality': {
                good: 'Perceptually similar',
                medium: 'Moderate differences',
                poor: 'Significant perceptual changes'
            },
            'Attribute Preservation': {
                good: 'Attributes well preserved',
                medium: 'Some attribute changes',
                poor: 'Significant attribute loss'
            },
            'Security Threshold': {
                good: 'Below security threshold',
                poor: 'Above security threshold'
            }
        };
        
        return descriptions[info.category]?.[status] || '';
    }
}