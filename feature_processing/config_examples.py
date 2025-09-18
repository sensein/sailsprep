"""
Configuration examples for the modular pipeline
Shows different use cases and how to customize the pipeline behavior
"""

from sailsprep.feature_processing.modular_pipeline import PipelineConfig, ModelConfig, FeatureConfig, ProcessingConfig, VisualizationConfig


def config_fast_processing():
    """Configuration optimized for speed - minimal features"""
    config = PipelineConfig()

    # Disable expensive features
    config.features.enable_face_features = False
    config.features.enable_lower_body_features = False
    config.features.enable_upper_body_features = True  # Keep only upper body
    config.features.feature_update_interval = 20  # Update less frequently

    # Relaxed thresholds for faster matching
    config.processing.detection_confidence_threshold = 0.6
    config.processing.combined_reid_threshold = 0.6

    # Minimal visualization
    config.visualization.enable_pose_drawing = False
    config.visualization.enable_bbox_drawing = True
    config.visualization.enable_id_labels = True

    return config


def config_high_accuracy():
    """Configuration optimized for accuracy - all features enabled"""
    config = PipelineConfig()

    # Enable all features
    config.features.enable_face_features = True
    config.features.enable_upper_body_features = True
    config.features.enable_lower_body_features = True
    config.features.feature_update_interval = 5  # Frequent updates

    # Strict thresholds for better precision
    config.processing.detection_confidence_threshold = 0.7
    config.processing.combined_reid_threshold = 0.8
    config.processing.face_reid_threshold = 0.8
    config.processing.upper_reid_threshold = 0.7
    config.processing.lower_reid_threshold = 0.65

    # Full visualization
    config.visualization.enable_pose_drawing = True
    config.visualization.enable_bbox_drawing = True
    config.visualization.enable_id_labels = True

    return config


def config_no_face_features():
    """Configuration without face features (privacy-preserving)"""
    config = PipelineConfig()

    # Disable face features completely
    config.features.enable_face_features = False
    config.features.enable_upper_body_features = True
    config.features.enable_lower_body_features = True

    # Adjust thresholds since no face features
    config.processing.combined_reid_threshold = 0.65
    config.processing.upper_reid_threshold = 0.7
    config.processing.lower_reid_threshold = 0.65

    return config


def config_batch_processing():
    """Configuration for batch processing multiple videos"""
    config = PipelineConfig()

    # Optimize for throughput
    config.processing.batch_size = 4  # Process multiple frames at once
    config.features.feature_update_interval = 15

    # Disable visualization for batch processing
    config.visualization.enable_visualization = False

    return config


def config_different_models():
    """Configuration with different model paths"""
    config = PipelineConfig()

    # Example: Use different detection model
    # config.models.detection_config = "path/to/different/detection/config.py"
    # config.models.detection_checkpoint = "path/to/different/detection/checkpoint.pth"

    # Example: Use different pose model
    # config.models.pose_config = "path/to/different/pose/config.py"
    # config.models.pose_checkpoint = "path/to/different/pose/checkpoint.pth"

    # Example: Use different device
    # config.models.device = 'cuda:1'

    return config


def config_crowded_scenes():
    """Configuration optimized for crowded scenes with many people"""
    config = PipelineConfig()

    # Stricter detection to reduce false positives
    config.processing.detection_confidence_threshold = 0.7
    config.processing.nms_threshold = 0.5  # More aggressive NMS

    # Longer tracking history for crowded scenes
    config.processing.max_lost_frames = 150  # Shorter to prevent ID switches

    # More frequent feature updates due to occlusions
    config.features.feature_update_interval = 8

    # Higher thresholds to prevent wrong associations
    config.processing.combined_reid_threshold = 0.75
    config.processing.face_reid_threshold = 0.8

    return config


def config_long_term_tracking():
    """Configuration optimized for long-term tracking"""
    config = PipelineConfig()

    # Keep tracks alive longer
    config.processing.max_lost_frames = 600  # 20 seconds at 30 FPS

    # More conservative feature updates to maintain consistency
    config.features.feature_update_interval = 20

    # Enable all features for better re-identification
    config.features.enable_face_features = True
    config.features.enable_upper_body_features = True
    config.features.enable_lower_body_features = True

    return config


def config_real_time():
    """Configuration for real-time processing"""
    config = PipelineConfig()

    # Minimal features for speed
    config.features.enable_face_features = False
    config.features.enable_lower_body_features = False
    config.features.feature_update_interval = 30

    # Lower quality thresholds for speed
    config.processing.detection_confidence_threshold = 0.5
    config.processing.combined_reid_threshold = 0.6

    # Minimal visualization
    config.visualization.enable_pose_drawing = False

    return config


def config_research_quality():
    """Configuration for research with maximum data extraction"""
    config = PipelineConfig()

    # All features enabled
    config.features.enable_face_features = True
    config.features.enable_upper_body_features = True
    config.features.enable_lower_body_features = True
    config.features.feature_update_interval = 1  # Extract every frame

    # Conservative thresholds to minimize errors
    config.processing.detection_confidence_threshold = 0.8
    config.processing.combined_reid_threshold = 0.85

    # Full visualization with detailed annotations
    config.visualization.enable_pose_drawing = True
    config.visualization.enable_bbox_drawing = True
    config.visualization.enable_id_labels = True

    return config


# Dictionary of all configurations for easy access
CONFIGS = {
    'fast': config_fast_processing,
    'accurate': config_high_accuracy,
    'no_face': config_no_face_features,
    'batch': config_batch_processing,
    'different_models': config_different_models,
    'crowded': config_crowded_scenes,
    'long_term': config_long_term_tracking,
    'real_time': config_real_time,
    'research': config_research_quality
}


def get_config(config_name: str) -> PipelineConfig:
    """Get a configuration by name"""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")

    return CONFIGS[config_name]()


def list_available_configs():
    """List all available configurations"""
    print("Available configurations:")
    for name, func in CONFIGS.items():
        print(f"  {name}: {func.__doc__.strip()}")


if __name__ == "__main__":
    # Example usage
    list_available_configs()

    # Get a specific configuration
    fast_config = get_config('fast')
    print(f"\nFast config face features enabled: {fast_config.features.enable_face_features}")

    accurate_config = get_config('accurate')
    print(f"Accurate config reid threshold: {accurate_config.processing.combined_reid_threshold}")