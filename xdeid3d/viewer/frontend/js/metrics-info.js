// Metric descriptions based on GUARD documentation
export const METRICS_INFO = {
    // Identity Distance Metrics
    COSINE_SIMILARITY: {
        name: "Cosine Similarity",
        category: "Identity Distance",
        description: "Measures the angle between face embeddings in identity space",
        interpretation: {
            good: "< 0.5 (successful de-identification)",
            medium: "0.5 - 0.7 (partial anonymization)",
            poor: "> 0.7 (identity preserved)"
        },
        details: "Range: 0 to 1. For de-identification, lower is better. Same person typically scores 0.8-0.9.",
        formula: "cos(θ) = (A·B) / (||A||×||B||)"
    },
    
    COSINE_DISTANCE: {
        name: "Cosine Distance",
        category: "Identity Distance",
        description: "Angular difference between face embeddings (1 - cosine similarity)",
        interpretation: {
            good: "> 0.5 (successful de-identification)",
            medium: "0.3 - 0.5 (partial anonymization)",
            poor: "< 0.3 (identity preserved)"
        },
        details: "Range: 0 to 1. Higher values indicate better anonymization. Complement of cosine similarity.",
        formula: "1 - cosine_similarity"
    },
    
    L2_DISTANCE: {
        name: "L2 (Euclidean) Distance",
        category: "Identity Distance",
        description: "Straight-line distance between face embeddings in identity space",
        interpretation: {
            good: "> 15 (successful de-identification)",
            medium: "10 - 15 (partial anonymization)",
            poor: "< 10 (identity preserved)"
        },
        details: "Range: 0 to ∞ (typically 0-30). Higher values indicate better anonymization. Actual implementation shows values 10-25.",
        formula: "√(Σ(a_i - b_i)²)"
    },
    
    // FAR Thresholds
    FAR_1E3: {
        name: "FAR@1e-3",
        category: "Security Threshold",
        description: "False Acceptance Rate at 0.1% - General applications",
        interpretation: {
            good: "Not matched (similarity < 0.7629)",
            poor: "Matched (similarity ≥ 0.7629)"
        },
        details: "1 in 1,000 false matches. Used for general face verification applications.",
        threshold: 0.7629
    },
    
    FAR_1E4: {
        name: "FAR@1e-4",
        category: "Security Threshold",
        description: "False Acceptance Rate at 0.01% - Enhanced security",
        interpretation: {
            good: "Not matched (similarity < 0.8227)",
            poor: "Matched (similarity ≥ 0.8227)"
        },
        details: "1 in 10,000 false matches. Used for payment verification and secure access.",
        threshold: 0.8227
    },
    
    FAR_1E5: {
        name: "FAR@1e-5",
        category: "Security Threshold",
        description: "False Acceptance Rate at 0.001% - High security",
        interpretation: {
            good: "Not matched (similarity < 0.8601)",
            poor: "Matched (similarity ≥ 0.8601)"
        },
        details: "1 in 100,000 false matches. Used for border control and critical security.",
        threshold: 0.8601
    },
    
    // Visual Quality Metrics
    LPIPS: {
        name: "LPIPS",
        category: "Perceptual Quality",
        description: "Learned Perceptual Image Patch Similarity - balances anonymization with visual quality",
        interpretation: {
            good: "0.3 - 0.7 (balanced anonymization)",
            medium: "0.2 - 0.3 or 0.7 - 0.8",
            poor: "< 0.2 (too similar) or > 0.8 (too different)"
        },
        details: "Range: 0 to 1. For anonymization, 0.3-0.7 provides good balance between identity change and quality.",
        formula: "Distance in VGG feature space"
    },
    
    PSNR: {
        name: "PSNR",
        category: "Image Quality",
        description: "Peak Signal-to-Noise Ratio - measures pixel-level reconstruction quality",
        interpretation: {
            good: "> 30 dB (high quality)",
            medium: "25 - 30 dB",
            poor: "< 25 dB (visible artifacts)"
        },
        details: "Range: 0 to ∞ dB. Higher is better. 30+ dB typically indicates good quality.",
        formula: "20 × log₁₀(MAX / √MSE)"
    },
    
    SSIM: {
        name: "SSIM",
        category: "Image Quality",
        description: "Structural Similarity Index - measures structural preservation",
        interpretation: {
            good: "> 0.9 (structure preserved)",
            medium: "0.7 - 0.9",
            poor: "< 0.7 (structure degraded)"
        },
        details: "Range: -1 to 1. Higher is better. Considers luminance, contrast, and structure.",
        formula: "Combines luminance, contrast, and structure comparison"
    },
    
    FID: {
        name: "FID",
        category: "Distribution Quality",
        description: "Fréchet Inception Distance - measures realism of generated faces",
        interpretation: {
            good: "< 50 (realistic)",
            medium: "50 - 100",
            poor: "> 100 (unrealistic)"
        },
        details: "Range: 0 to ∞. Lower is better. Compares feature distributions of real vs generated.",
        formula: "||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r × Σ_g))"
    },
    
    // Attribute Preservation
    GENDER_PRESERVATION: {
        name: "Gender Preservation",
        category: "Attribute Preservation",
        description: "Measures if gender is maintained after anonymization",
        interpretation: {
            good: "> 95% accuracy",
            medium: "85 - 95%",
            poor: "< 85%"
        },
        details: "Percentage of correctly preserved gender. Important for maintaining demographic attributes."
    },
    
    EXPRESSION_PRESERVATION: {
        name: "Expression Preservation",
        category: "Attribute Preservation",
        description: "Measures if facial expression is maintained",
        interpretation: {
            good: "< 0.1 deviation",
            medium: "0.1 - 0.3",
            poor: "> 0.3"
        },
        details: "Based on 3DMM expression parameters. Lower deviation means better preservation."
    },
    
    POSE_PRESERVATION: {
        name: "Pose Preservation",
        category: "Attribute Preservation",
        description: "Measures if head pose/orientation is maintained",
        interpretation: {
            good: "< 5° error",
            medium: "5° - 15°",
            poor: "> 15°"
        },
        details: "Angular error in head pose. Important for maintaining natural appearance."
    }
};

// Additional metrics from video anonymization guide
const ADDITIONAL_METRICS = {
    // ArcFace-specific metrics
    ARCFACE_COSINE_SIMILARITY: {
        name: "ArcFace Cosine Similarity",
        category: "Identity Distance",
        description: "Cosine similarity using ArcFace face recognition model",
        interpretation: {
            good: "< 0.5 (successful de-identification)",
            medium: "0.5 - 0.7 (partial anonymization)",
            poor: "> 0.7 (identity preserved)"
        },
        details: "Range: 0 to 1. ArcFace is state-of-the-art for face recognition. FAR@1e-3 threshold: 0.7629.",
        formula: "cos(θ) = (A·B) / (||A||×||B||)"
    },
    
    ARCFACE_COSINE_DISTANCE: {
        name: "ArcFace Cosine Distance",
        category: "Identity Distance",
        description: "Angular difference using ArcFace embeddings (1 - cosine similarity)",
        interpretation: {
            good: "> 0.5 (successful de-identification)",
            medium: "0.3 - 0.5 (partial anonymization)",
            poor: "< 0.3 (identity preserved)"
        },
        details: "Range: 0 to 1. Higher values indicate better anonymization. Complement of similarity score.",
        formula: "1 - cosine_similarity"
    },
    
    ARCFACE_L2_DISTANCE: {
        name: "ArcFace L2 Distance",
        category: "Identity Distance",
        description: "Euclidean distance between ArcFace face embeddings",
        interpretation: {
            good: "> 15 (successful de-identification)",
            medium: "10 - 15 (partial anonymization)",
            poor: "< 10 (identity preserved)"
        },
        details: "Range: 0 to 30 typically. Measures distance in 512-dimensional embedding space. Higher is better.",
        formula: "√(Σ(a_i - b_i)²)"
    },
    L1_DISTANCE: {
        name: "L1 Distance",
        category: "Image Quality",
        description: "Average absolute pixel difference - balances anonymization with quality",
        interpretation: {
            good: "0.1 - 0.3 (balanced change)",
            medium: "0.05 - 0.1 or 0.3 - 0.4",
            poor: "< 0.05 (weak anonymization) or > 0.4 (excessive change)"
        },
        details: "Range: 0 to 1. For anonymization, 0.1-0.3 indicates good balance between identity change and quality.",
        formula: "mean(|pixel_orig - pixel_anon|)"
    },
    
    MSE: {
        name: "MSE (Mean Squared Error)",
        category: "Image Quality",
        description: "Average squared pixel difference - balances anonymization with quality",
        interpretation: {
            good: "0.05 - 0.15 (balanced change)",
            medium: "0.02 - 0.05 or 0.15 - 0.25",
            poor: "< 0.02 (weak anonymization) or > 0.25 (poor quality)"
        },
        details: "Range: 0 to 1. Zero means no anonymization. For good de-identification, expect 0.05-0.15.",
        formula: "mean((pixel_orig - pixel_anon)²)"
    },
    
    TEMPORAL_IDENTITY_CONSISTENCY: {
        name: "Temporal Identity Consistency",
        category: "Temporal Consistency",
        description: "Stability of anonymized identity between consecutive frames",
        interpretation: {
            good: "> 0.9 (stable anonymization)",
            medium: "0.8 - 0.9 (some variation)",
            poor: "< 0.8 (flickering/unstable)"
        },
        details: "Range: 0 to 1. High values prevent flickering while maintaining anonymization. Critical for video quality.",
        formula: "cosine_similarity(anon_frame_t, anon_frame_t+1)"
    },
    
    TEMPORAL_VISUAL_SMOOTHNESS: {
        name: "Temporal Visual Smoothness",
        category: "Temporal Consistency",
        description: "Pixel-level consistency between consecutive frames",
        interpretation: {
            good: "> 0.8 (smooth)",
            medium: "0.6 - 0.8",
            poor: "< 0.6 (jerky)"
        },
        details: "Range: 0 to 1 (1 = perfectly smooth). Higher values indicate smoother transitions.",
        formula: "1 - min(MSE(frame_t, frame_t+1), 1)"
    },
    
    IDENTITY_DRIFT: {
        name: "Identity Drift",
        category: "Temporal Consistency",
        description: "Standard deviation of anonymized embeddings over time",
        interpretation: {
            good: "< 0.05 (minimal drift)",
            medium: "0.05 - 0.1",
            poor: "> 0.1 (high drift)"
        },
        details: "Range: 0 to ∞ (0 = no drift). Measures identity stability over sliding windows.",
        formula: "std(embeddings) over window"
    },
    
    ANONYMIZATION_COVERAGE: {
        name: "Anonymization Coverage",
        category: "Detection Quality",
        description: "Percentage of frames where faces were detected and anonymized",
        interpretation: {
            good: "> 95% (reliable)",
            medium: "90 - 95%",
            poor: "< 90% (many missed)"
        },
        details: "Range: 0 to 100%. Higher coverage indicates more reliable face detection.",
        formula: "(anonymized_frames / total_frames) × 100"
    }
};

// Merge additional metrics
Object.assign(METRICS_INFO, ADDITIONAL_METRICS);

// Helper function to get metric info by key
export function getMetricInfo(metricKey) {
    if (!metricKey) {
        return {
            name: "Unknown Metric",
            category: "Unknown",
            description: "No metric information available",
            interpretation: {
                good: "N/A",
                medium: "N/A",
                poor: "N/A"
            },
            details: "No metric key provided"
        };
    }
    
    // Convert common variations to standard keys
    const keyMap = {
        'cosine': 'COSINE_SIMILARITY',
        'cosine_sim': 'COSINE_SIMILARITY',
        'cosine_similarity': 'COSINE_SIMILARITY',
        'cosine_dist': 'COSINE_DISTANCE',
        'cosine_distance': 'COSINE_DISTANCE',
        'l2': 'L2_DISTANCE',
        'l2_dist': 'L2_DISTANCE',
        'l2_distance': 'L2_DISTANCE',
        'l1': 'L1_DISTANCE',
        'l1_dist': 'L1_DISTANCE',
        'l1_distance': 'L1_DISTANCE',
        'far_1e3': 'FAR_1E3',
        'far_1e4': 'FAR_1E4',
        'far_1e5': 'FAR_1E5',
        'lpips': 'LPIPS',
        'lpips_distance': 'LPIPS',
        'psnr': 'PSNR',
        'ssim': 'SSIM',
        'fid': 'FID',
        'mse': 'MSE',
        'gender': 'GENDER_PRESERVATION',
        'expression': 'EXPRESSION_PRESERVATION',
        'pose': 'POSE_PRESERVATION',
        'temporal_identity_consistency': 'TEMPORAL_IDENTITY_CONSISTENCY',
        'temporal_visual_smoothness': 'TEMPORAL_VISUAL_SMOOTHNESS',
        'identity_drift': 'IDENTITY_DRIFT',
        'anonymization_coverage': 'ANONYMIZATION_COVERAGE',
        'expression_l2_distance': 'EXPRESSION_PRESERVATION',
        'pose_l2_distance': 'POSE_PRESERVATION',
        'shape_landmarks_l2_distance': 'EXPRESSION_PRESERVATION',
        // ArcFace specific mappings
        'arcface_cosine_similarity': 'ARCFACE_COSINE_SIMILARITY',
        'arcface_cosine_distance': 'ARCFACE_COSINE_DISTANCE',
        'arcface_l2_distance': 'ARCFACE_L2_DISTANCE',
        'mesh_arcface_cosine_similarity': 'ARCFACE_COSINE_SIMILARITY',
        'mesh_arcface_cosine_distance': 'ARCFACE_COSINE_DISTANCE',
        'mesh_arcface_l2_distance': 'ARCFACE_L2_DISTANCE',
        'mesh_l1_distance': 'L1_DISTANCE',
        'mesh_lpips_distance': 'LPIPS',
        'mesh_psnr': 'PSNR',
        'mesh_mse': 'MSE',
        'mesh_temporal_identity_consistency': 'TEMPORAL_IDENTITY_CONSISTENCY',
        'mesh_temporal_visual_smoothness': 'TEMPORAL_VISUAL_SMOOTHNESS'
    };
    
    const normalizedKey = metricKey.toLowerCase().replace(/[^a-z0-9_]/g, '_');
    const mappedKey = keyMap[normalizedKey] || metricKey.toUpperCase().replace(/-/g, '_');
    
    // Try to find the metric info directly
    if (METRICS_INFO[mappedKey]) {
        return METRICS_INFO[mappedKey];
    }
    
    // Check if it's a known metric pattern that wasn't mapped
    const lowerKey = metricKey.toLowerCase();
    
    // Handle ArcFace/CosFace/FaceNet variations
    if (lowerKey.includes('arcface') || lowerKey.includes('cosface') || lowerKey.includes('facenet')) {
        if (lowerKey.includes('cosine') && !lowerKey.includes('dist')) {
            return {
                name: `${metricKey.split('_')[0].toUpperCase()} Cosine Similarity`,
                category: "Identity Distance",
                description: "Similarity between face embeddings using specific face recognition model",
                interpretation: {
                    good: "< 0.3 (faces don't match)",
                    medium: "0.3 - 0.6",
                    poor: "> 0.6 (faces still similar)"
                },
                details: "Range: 0 to 1. Lower is better for anonymization. Same person typically scores 0.8-0.9.",
                formula: "cos(θ) = (A·B) / (||A||×||B||)"
            };
        } else if (lowerKey.includes('dist')) {
            return {
                name: `${metricKey.split('_')[0].toUpperCase()} Cosine Distance`,
                category: "Identity Distance",
                description: "Angular difference between face embeddings",
                interpretation: {
                    good: "> 0.7 (large difference)",
                    medium: "0.4 - 0.7",
                    poor: "< 0.4 (small difference)"
                },
                details: "Range: 0 to 1. Higher is better for anonymization.",
                formula: "1 - cosine_similarity"
            };
        } else if (lowerKey.includes('l2')) {
            return {
                name: `${metricKey.split('_')[0].toUpperCase()} L2 Distance`,
                category: "Identity Distance",
                description: "Euclidean distance between face embeddings",
                interpretation: {
                    good: "> 20 (far apart)",
                    medium: "15 - 20",
                    poor: "< 15 (still close)"
                },
                details: "Range: 0 to ∞. Higher is better. More sensitive than cosine to all embedding changes.",
                formula: "√(Σ(a_i - b_i)²)"
            };
        } else if (lowerKey.includes('far')) {
            // Extract FAR level
            const farMatch = lowerKey.match(/far[@_]?1e[-_]?(\d+)/);
            if (farMatch) {
                const level = farMatch[1];
                const thresholds = { '3': '0.7629', '4': '0.8227', '5': '0.8601' };
                const threshold = thresholds[level] || '0.7629';
                const percentage = level === '3' ? '0.1%' : level === '4' ? '0.01%' : '0.001%';
                
                return {
                    name: `${metricKey.split('_')[0].toUpperCase()} FAR@1e-${level}`,
                    category: "Security Threshold",
                    description: `False Acceptance Rate at ${percentage} - ${level === '3' ? 'General' : level === '4' ? 'Enhanced' : 'High'} security`,
                    interpretation: {
                        good: `Not matched (similarity < ${threshold})`,
                        poor: `Matched (similarity ≥ ${threshold})`
                    },
                    details: `1 in ${Math.pow(10, parseInt(level))} false matches. Used for ${level === '3' ? 'general applications' : level === '4' ? 'payment verification' : 'border control'}.`,
                    threshold: parseFloat(threshold)
                };
            }
        }
    }
    
    // Handle defense-related metrics
    if (lowerKey.includes('defense') || lowerKey.includes('reconstruction')) {
        return {
            name: metricKey.split(/[_-]/).map(word => 
                word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
            ).join(' '),
            category: "Defense Effectiveness",
            description: "Measures effectiveness of defense against reconstruction attacks",
            interpretation: {
                good: "> 0.05 reduction in similarity",
                medium: "0.02 - 0.05 reduction",
                poor: "< 0.02 reduction"
            },
            details: "Compares identity similarity before and after defense. Higher reduction is better."
        };
    }
    
    // Handle attribute metrics
    if (lowerKey.includes('gender') || lowerKey.includes('ethnicity') || lowerKey.includes('age')) {
        const attribute = lowerKey.includes('gender') ? 'Gender' : 
                         lowerKey.includes('ethnicity') ? 'Ethnicity' : 'Age';
        return {
            name: `${attribute} Preservation`,
            category: "Attribute Preservation",
            description: `Measures if ${attribute.toLowerCase()} is maintained after anonymization`,
            interpretation: {
                good: "> 95% accuracy",
                medium: "85 - 95%",
                poor: "< 85%"
            },
            details: `Percentage of correctly preserved ${attribute.toLowerCase()}. Important for maintaining demographic attributes.`
        };
    }
    
    // Handle shape_landmarks_l2_distance specifically
    if (lowerKey.includes('shape_landmarks')) {
        return {
            name: "Shape Landmarks L2 Distance",
            category: "Attribute Preservation",
            description: "Measures difference in facial landmark positions",
            interpretation: {
                good: "< 10.0 (shape well preserved)",
                medium: "10.0 - 20.0",
                poor: "> 20.0 (significant shape changes)"
            },
            details: "Range: 0 to ∞ (0 = identical shape). Lower values indicate better preservation of facial structure.",
            formula: "√(Σ(landmark_orig - landmark_anon)²)"
        };
    }
    
    // Final fallback with better defaults based on common patterns
    const finalName = metricKey.split(/[_-]/).map(word => {
        // Special handling for known abbreviations
        const specialWords = {
            'l1': 'L1',
            'l2': 'L2',
            'mse': 'MSE',
            'psnr': 'PSNR',
            'ssim': 'SSIM',
            'lpips': 'LPIPS',
            'fid': 'FID',
            'far': 'FAR'
        };
        return specialWords[word.toLowerCase()] || (word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());
    }).join(' ');
    
    // Provide more specific defaults based on metric name patterns
    let category = "Evaluation Metric";
    let description = `Measures ${finalName.toLowerCase()} in face anonymization`;
    let interpretation = {
        good: "Optimal range",
        medium: "Acceptable range",
        poor: "Suboptimal range"
    };
    let details = "";
    
    // Better categorization based on metric name
    if (lowerKey.includes('identity') || lowerKey.includes('cosine') || lowerKey.includes('l2')) {
        category = "Identity Distance";
        description = "Measures identity similarity/distance after anonymization";
        details = "Lower similarity or higher distance indicates better anonymization.";
    } else if (lowerKey.includes('temporal')) {
        category = "Temporal Consistency";
        description = "Measures consistency across video frames";
        details = "Higher values indicate more stable anonymization over time.";
    } else if (lowerKey.includes('psnr') || lowerKey.includes('ssim') || lowerKey.includes('mse')) {
        category = "Image Quality";
        description = "Measures visual quality preservation";
        details = "Balance needed between quality and anonymization effectiveness.";
    } else if (lowerKey.includes('lpips') || lowerKey.includes('perceptual')) {
        category = "Perceptual Quality";
        description = "Measures perceptual similarity using deep features";
        details = "Based on human perception rather than pixel differences.";
    }
    
    return {
        name: finalName,
        category: category,
        description: description,
        interpretation: interpretation,
        details: details
    };
}