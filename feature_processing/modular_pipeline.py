import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np
import os
import cv2
import ffmpeg
from tqdm import tqdm
import torch
import time
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from facenet_pytorch import MTCNN
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import math
import signal
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod

from deepface import DeepFace
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector, init_detector
from filterpy.kalman import KalmanFilter


# ================== CONFIGURATION SYSTEM ==================

@dataclass
class ModelConfig:
    """Configuration for models"""
    # Detection model
    detection_config: str = "projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py"
    detection_checkpoint: str = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

    # Pose model
    pose_config: str = 'configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py'
    pose_checkpoint: str = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'

    device: str = 'cuda:0'
    mmpose_path: str = "sailsprep/feature_processing/mmpose"

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    enable_face_features: bool = True
    enable_upper_body_features: bool = True
    enable_lower_body_features: bool = True

    # Face feature settings
    face_confidence_threshold: float = 0.75
    face_min_size: int = 40

    # Body feature settings
    body_min_height: int = 50
    body_min_width: int = 30

    # Feature update frequency
    feature_update_interval: int = 10

@dataclass
class ProcessingConfig:
    """Configuration for processing parameters"""
    # Detection settings
    detection_confidence_threshold: float = 0.5
    nms_threshold: float = 0.7
    batch_size: int = 1

    # Tracking settings
    iou_threshold: float = 0.3
    motion_confidence_threshold: float = 0.5
    max_lost_frames: int = 300

    # Re-identification thresholds
    face_reid_threshold: float = 0.75
    upper_reid_threshold: float = 0.65
    lower_reid_threshold: float = 0.6
    combined_reid_threshold: float = 0.7

@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    enable_visualization: bool = True
    enable_pose_drawing: bool = True
    enable_bbox_drawing: bool = True
    enable_id_labels: bool = True

    # Visualization parameters
    keypoint_threshold: float = 0.3
    radius: int = 3
    line_width: int = 1

@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    models: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


# ================== DETECTION MODULE ==================

class DetectionModule:
    """Handles person detection with batching support"""

    def __init__(self, config: ModelConfig, processing_config: ProcessingConfig):
        self.config = config
        self.processing_config = processing_config
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        """Initialize detection model"""
        det_config = os.path.join(self.config.mmpose_path, self.config.detection_config)
        self.detector = init_detector(
            det_config,
            self.config.detection_checkpoint,
            device=self.config.device
        )

    def detect_single(self, frame: np.ndarray) -> np.ndarray:
        """Detect persons in single frame"""
        scope = self.detector.cfg.get('default_scope', 'mmdet')
        if scope is not None:
            init_default_scope(scope)

        detect_result = inference_detector(self.detector, frame)
        return self._process_detection_result(detect_result)

    def detect_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Detect persons in batch of frames"""
        # For now, process sequentially - can be enhanced for true batching
        results = []
        for frame in frames:
            results.append(self.detect_single(frame))
        return results

    def _process_detection_result(self, detect_result) -> np.ndarray:
        """Process detection result and apply filtering"""
        pred_instance = detect_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)

        # Filter by confidence and class (person=0)
        bboxes = bboxes[np.logical_and(
            pred_instance.labels == 0,
            pred_instance.scores > self.processing_config.detection_confidence_threshold
        )]

        # Apply NMS
        if len(bboxes) > 0:
            bboxes = bboxes[nms(bboxes, self.processing_config.nms_threshold)][:, :4]

        return bboxes


# ================== POSE ESTIMATION MODULE ==================

class PoseEstimationModule:
    """Handles pose estimation with batching support"""

    def __init__(self, config: ModelConfig, visualization_config: VisualizationConfig):
        self.config = config
        self.vis_config = visualization_config
        self.pose_estimator = None
        self.visualizer = None
        self._init_pose_estimator()

    def _init_pose_estimator(self):
        """Initialize pose estimation model"""
        pose_config = os.path.join(self.config.mmpose_path, self.config.pose_config)
        self.pose_estimator = init_pose_estimator(
            pose_config,
            self.config.pose_checkpoint,
            device=self.config.device,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=True)))
        )

        # Setup visualizer
        if self.vis_config.enable_visualization:
            self.pose_estimator.cfg.visualizer.radius = self.vis_config.radius
            self.pose_estimator.cfg.visualizer.line_width = self.vis_config.line_width
            self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
            self.visualizer.set_dataset_meta(self.pose_estimator.dataset_meta)

    def estimate_single(self, frame: np.ndarray, bboxes: np.ndarray) -> List:
        """Estimate poses for single frame"""
        if len(bboxes) == 0:
            return []

        return inference_topdown(self.pose_estimator, frame, bboxes)

    def estimate_batch(self, frames: List[np.ndarray], bboxes_list: List[np.ndarray]) -> List[List]:
        """Estimate poses for batch of frames"""
        results = []
        for frame, bboxes in zip(frames, bboxes_list):
            results.append(self.estimate_single(frame, bboxes))
        return results


# ================== FEATURE EXTRACTION MODULE ==================

class FeatureExtractor:
    """Base class for feature extractors"""

    @abstractmethod
    def extract(self, roi: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        pass

class FaceFeatureExtractor(FeatureExtractor):
    """Extracts face features using DeepFace"""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.mtcnn = MTCNN(
            keep_all=True,
            device='cuda:0',
            post_process=False,
            min_face_size=config.face_min_size
        )

    def extract(self, face_roi: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """Extract face embedding"""
        if not self.config.enable_face_features:
            return None

        try:
            # Validate with MTCNN
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            boxes, probs = self.mtcnn.detect(face_rgb)

            if boxes is None or probs is None or len(boxes) == 0:
                return None

            best_prob = float(np.max(probs))
            if best_prob < self.config.face_confidence_threshold:
                return None

            # Extract DeepFace embedding
            face_resized = cv2.resize(face_roi, (112, 112))
            embedding_result = DeepFace.represent(
                face_resized,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='skip'
            )
            embedding = np.array(embedding_result[0]['embedding'])

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                return embedding

        except:
            pass

        return None

class BodyFeatureExtractor(FeatureExtractor):
    """Extracts body appearance features"""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def extract(self, roi: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """Extract body appearance features"""
        try:
            if roi.shape[0] < self.config.body_min_height or roi.shape[1] < self.config.body_min_width:
                return None

            features = []

            # HSV color features
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])

            for hist in [h_hist, s_hist, v_hist]:
                hist_norm = cv2.normalize(hist, hist).flatten()
                features.append(hist_norm)

            # Texture features (LBP)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray_roi, (48, 64))
            lbp_hist = self._compute_lbp_histogram(gray_resized)
            features.append(lbp_hist)

            # Edge features
            edges = cv2.Canny(gray_resized, 50, 150)
            edge_hist, _ = np.histogram(edges.sum(axis=1), bins=12)
            edge_hist = edge_hist / (edge_hist.sum() + 1e-7)
            features.append(edge_hist)

            # Combine all features
            combined_feature = np.concatenate(features)

            # L2 normalize
            norm = np.linalg.norm(combined_feature)
            if norm > 0:
                return combined_feature / norm

        except:
            pass

        return None

    def _compute_lbp_histogram(self, gray_image: np.ndarray) -> np.ndarray:
        """Compute LBP histogram"""
        try:
            rows, cols = gray_image.shape
            lbp_image = np.zeros_like(gray_image)

            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    center = gray_image[i, j]
                    code = 0

                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                        gray_image[i+1, j-1], gray_image[i, j-1]
                    ]

                    for k, neighbor in enumerate(neighbors):
                        if neighbor > center:
                            code |= (1 << k)

                    lbp_image[i, j] = code

            hist, _ = np.histogram(lbp_image, bins=24, range=(0, 256))
            return hist / (hist.sum() + 1e-7)
        except:
            return np.zeros(24)

class RegionExtractor:
    """Extracts different body regions from frames"""

    @staticmethod
    def extract_face_region(frame: np.ndarray, keypoints: np.ndarray, bbox: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract face region using head keypoints"""
        try:
            kpts = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
            if len(kpts.shape) == 3:
                kpts = kpts[0]

            # Get head keypoints (nose, eyes, ears)
            head_keypoints = [0, 1, 2, 3, 4]
            head_points = []
            for idx in head_keypoints:
                if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                    head_points.append(kpts[idx][:2])

            if len(head_points) >= 2:
                head_points = np.array(head_points)
                x_min, y_min = np.min(head_points, axis=0)
                x_max, y_max = np.max(head_points, axis=0)

                padding = 25
                face_x1 = max(0, int(x_min - padding))
                face_y1 = max(0, int(y_min - padding))
                face_x2 = min(frame.shape[1], int(x_max + padding))
                face_y2 = min(frame.shape[0], int(y_max + padding))
            else:
                # Fallback to upper bbox region
                x1, y1, x2, y2 = bbox.astype(int)
                face_h = int((y2 - y1) * 0.35)
                face_x1, face_y1 = x1, y1
                face_x2, face_y2 = x2, y1 + face_h

            if face_x2 <= face_x1 or face_y2 <= face_y1:
                return None

            face_roi = frame[face_y1:face_y2, face_x1:face_x2]

            if face_roi.shape[0] < 40 or face_roi.shape[1] < 30:
                return None

            return face_roi, (face_x1, face_y1, face_x2, face_y2)
        except:
            return None

    @staticmethod
    def extract_upper_body_region(frame: np.ndarray, keypoints: np.ndarray, bbox: np.ndarray, pose_type: str) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract upper body region"""
        try:
            kpts = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
            if len(kpts.shape) == 3:
                kpts = kpts[0]

            # Get neck point from shoulders
            neck_point = None
            left_shoulder = kpts[5] if len(kpts) > 5 and len(kpts[5]) >= 3 and kpts[5][2] > 0.3 else None
            right_shoulder = kpts[6] if len(kpts) > 6 and len(kpts[6]) >= 3 and kpts[6][2] > 0.3 else None

            if left_shoulder is not None and right_shoulder is not None:
                neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
                neck_y = (left_shoulder[1] + right_shoulder[1]) / 2 - 15
                neck_point = np.array([neck_x, neck_y])

            # Get hip points
            hip_points = []
            for idx in [11, 12]:  # left_hip, right_hip
                if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                    hip_points.append(kpts[idx][:2])

            if neck_point is not None and hip_points:
                hip_center = np.mean(hip_points, axis=0)
                upper_y1 = int(neck_point[1])
                upper_y2 = int(hip_center[1])

                # Use shoulders for width
                if left_shoulder is not None and right_shoulder is not None:
                    x_min = min(left_shoulder[0], right_shoulder[0])
                    x_max = max(left_shoulder[0], right_shoulder[0])
                    padding = 20
                    upper_x1 = max(0, int(x_min - padding))
                    upper_x2 = min(frame.shape[1], int(x_max + padding))
                else:
                    padding = 60
                    upper_x1 = max(0, int(neck_point[0] - padding))
                    upper_x2 = min(frame.shape[1], int(neck_point[0] + padding))
            else:
                # Fallback to bbox-based region
                x1, y1, x2, y2 = bbox.astype(int)

                if pose_type == "sitting":
                    upper_y1 = y1 + int((y2 - y1) * 0.1)
                    upper_y2 = y1 + int((y2 - y1) * 0.75)
                elif pose_type == "lying":
                    upper_y1 = y1 + int((y2 - y1) * 0.2)
                    upper_y2 = y1 + int((y2 - y1) * 0.8)
                else:  # standing
                    upper_y1 = y1 + int((y2 - y1) * 0.15)
                    upper_y2 = y1 + int((y2 - y1) * 0.65)

                upper_x1 = x1 + int((x2 - x1) * 0.1)
                upper_x2 = x2 - int((x2 - x1) * 0.1)

            # Ensure valid region
            upper_y1 = max(0, min(upper_y1, frame.shape[0]))
            upper_y2 = max(upper_y1, min(upper_y2, frame.shape[0]))
            upper_x1 = max(0, min(upper_x1, frame.shape[1]))
            upper_x2 = max(upper_x1, min(upper_x2, frame.shape[1]))

            if upper_y2 <= upper_y1 or upper_x2 <= upper_x1:
                return None

            upper_roi = frame[upper_y1:upper_y2, upper_x1:upper_x2]

            if upper_roi.shape[0] < 50 or upper_roi.shape[1] < 30:
                return None

            return upper_roi, (upper_x1, upper_y1, upper_x2, upper_y2)
        except:
            return None

    @staticmethod
    def extract_lower_body_region(frame: np.ndarray, keypoints: np.ndarray, bbox: np.ndarray, pose_type: str) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract lower body region"""
        if pose_type == "lying":
            return None

        try:
            kpts = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
            if len(kpts.shape) == 3:
                kpts = kpts[0]

            # Get hip points
            hip_points = []
            for idx in [11, 12]:  # left_hip, right_hip
                if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                    hip_points.append(kpts[idx][:2])

            # Get ankle points
            ankle_points = []
            for idx in [15, 16]:  # left_ankle, right_ankle
                if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                    ankle_points.append(kpts[idx][:2])

            if hip_points and ankle_points:
                hip_center = np.mean(hip_points, axis=0)
                ankle_center = np.mean(ankle_points, axis=0)

                lower_y1 = int(hip_center[1])
                lower_y2 = int(ankle_center[1]) + 20

                all_points = hip_points + ankle_points
                all_points = np.array(all_points)
                x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
                padding = 15
                lower_x1 = max(0, int(x_min - padding))
                lower_x2 = min(frame.shape[1], int(x_max + padding))
            else:
                # Fallback to bbox-based region
                x1, y1, x2, y2 = bbox.astype(int)

                if pose_type == "sitting":
                    lower_y1 = y1 + int((y2 - y1) * 0.6)
                    lower_y2 = y2
                else:  # standing
                    lower_y1 = y1 + int((y2 - y1) * 0.55)
                    lower_y2 = y2

                lower_x1 = x1 + int((x2 - x1) * 0.15)
                lower_x2 = x2 - int((x2 - x1) * 0.15)

            # Ensure valid region
            lower_y1 = max(0, min(lower_y1, frame.shape[0]))
            lower_y2 = max(lower_y1, min(lower_y2, frame.shape[0]))
            lower_x1 = max(0, min(lower_x1, frame.shape[1]))
            lower_x2 = max(lower_x1, min(lower_x2, frame.shape[1]))

            if lower_y2 <= lower_y1 or lower_x2 <= lower_x1:
                return None

            lower_roi = frame[lower_y1:lower_y2, lower_x1:lower_x2]

            if lower_roi.shape[0] < 40 or lower_roi.shape[1] < 25:
                return None

            return lower_roi, (lower_x1, lower_y1, lower_x2, lower_y2)
        except:
            return None

class FeatureExtractionModule:
    """Main feature extraction module coordinating all extractors"""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.face_extractor = FaceFeatureExtractor(config) if config.enable_face_features else None
        self.body_extractor = BodyFeatureExtractor(config)
        self.region_extractor = RegionExtractor()

    def extract_features(self, frame: np.ndarray, pose_results: List, frame_count: int) -> List[Dict]:
        """Extract features for all detections in frame"""
        detections = []

        for pose in pose_results:
            detection = self._create_detection(frame, pose, frame_count)
            if detection:
                detections.append(detection)

        return detections

    def _create_detection(self, frame: np.ndarray, pose, frame_count: int) -> Optional[Dict]:
        """Create detection with features from pose result"""
        if not hasattr(pose.pred_instances, 'bboxes') or len(pose.pred_instances.bboxes) == 0:
            return None

        bbox = pose.pred_instances.bboxes[0]
        if hasattr(bbox, 'cpu'):
            bbox = bbox.cpu().numpy()

        if len(bbox) < 4 or bbox[2] - bbox[0] < 50 or bbox[3] - bbox[1] < 100:
            return None

        keypoints = pose.pred_instances.keypoints[0]
        confidence = float(bbox[4]) if len(bbox) > 4 else 1.0
        pose_type = self._determine_pose_type(keypoints)

        # Extract features
        face_feature = None
        upper_feature = None
        lower_feature = None

        if frame_count % self.config.feature_update_interval == 0:
            # Face feature
            if self.config.enable_face_features:
                face_result = self.region_extractor.extract_face_region(frame, keypoints, bbox[:4])
                if face_result:
                    face_roi, _ = face_result
                    face_feature = self.face_extractor.extract(face_roi)

            # Lower body feature
            if self.config.enable_lower_body_features:
                lower_result = self.region_extractor.extract_lower_body_region(frame, keypoints, bbox[:4], pose_type)
                if lower_result:
                    lower_roi, _ = lower_result
                    lower_feature = self.body_extractor.extract(lower_roi)

        # Always extract upper body if enabled
        if self.config.enable_upper_body_features:
            upper_result = self.region_extractor.extract_upper_body_region(frame, keypoints, bbox[:4], pose_type)
            if upper_result:
                upper_roi, _ = upper_result
                upper_feature = self.body_extractor.extract(upper_roi)

        return {
            'bbox': bbox[:4],
            'keypoints': keypoints,
            'confidence': confidence,
            'pose_type': pose_type,
            'face_feature': face_feature,
            'upper_feature': upper_feature,
            'lower_feature': lower_feature
        }

    def _determine_pose_type(self, keypoints) -> str:
        """Determine pose type from keypoints"""
        try:
            kpts = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
            if len(kpts.shape) == 3:
                kpts = kpts[0]

            if len(kpts) < 17:
                return "standing"

            # Get key points with confidence > 0.3
            visible_points = {}
            for name, idx in [('left_hip', 11), ('right_hip', 12),
                            ('left_knee', 13), ('right_knee', 14),
                            ('left_ankle', 15), ('right_ankle', 16)]:
                if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                    visible_points[name] = kpts[idx][:2]

            if len(visible_points) < 3:
                return "standing"

            # Calculate hip-to-ankle distance
            hip_y = []
            ankle_y = []
            for hip in ['left_hip', 'right_hip']:
                if hip in visible_points:
                    hip_y.append(visible_points[hip][1])
            for ankle in ['left_ankle', 'right_ankle']:
                if ankle in visible_points:
                    ankle_y.append(visible_points[ankle][1])

            if hip_y and ankle_y:
                hip_ankle_dist = abs(np.mean(ankle_y) - np.mean(hip_y))

                if hip_ankle_dist < 50:
                    return "lying"
                elif hip_ankle_dist < 120:
                    return "sitting"
                else:
                    return "standing"

            return "standing"
        except:
            return "standing"


# ================== TRACKING MODULE ==================

class TrackingModule:
    """Handles multi-person tracking with re-identification"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.frame_count = 0
        self.next_track_id = 1
        self.active_tracks = {}
        self.lost_tracks = {}
        self.person_profiles = {}

    def update(self, detections: List[Dict]) -> Dict[int, int]:
        """Update tracking system with new detections"""
        self.frame_count += 1

        # 1. Motion-based matching
        motion_matches = self._match_with_motion(detections)
        final_matches = {}

        # Update matched tracks
        for det_idx, track_id in motion_matches.items():
            detection = detections[det_idx]
            track = self.active_tracks[track_id]

            # Update Kalman filter
            self._update_kalman_filter(track['kalman'], detection['bbox'])

            # Update track
            track['detections'].append(detection)
            track['last_seen'] = self.frame_count
            track['lost_frames'] = 0
            track['missed_updates'] = 0

            # Update profile
            profile = self.person_profiles.get(track_id)
            if profile:
                self._update_person_profile(profile, detection)

            final_matches[det_idx] = track_id

        # 2. Appearance-based matching
        unmatched_detections = [(i, det) for i, det in enumerate(detections) if i not in motion_matches]
        appearance_matches = self._match_with_appearance(unmatched_detections)

        for det_idx, (track_id, match_type) in appearance_matches.items():
            detection = detections[det_idx]
            track = self.active_tracks[track_id]

            self._update_kalman_filter(track['kalman'], detection['bbox'])
            track['detections'].append(detection)
            track['last_seen'] = self.frame_count
            track['lost_frames'] = 0
            track['missed_updates'] = 0

            profile = self.person_profiles.get(track_id)
            if profile:
                self._update_person_profile(profile, detection)

            final_matches[det_idx] = track_id

        # Update unmatched list
        unmatched_detections = [(i, det) for i, det in unmatched_detections if i not in appearance_matches]

        # 3. Re-identification with lost tracks
        reid_matches = self._match_with_lost_tracks(unmatched_detections)

        for det_idx, (track_id, match_type) in reid_matches.items():
            detection = detections[det_idx]

            # Reactivate lost track
            reactivated_track = self.lost_tracks.pop(track_id)
            reactivated_track['kalman'] = self._create_kalman_filter(detection['bbox'])
            reactivated_track['detections'].append(detection)
            reactivated_track['last_seen'] = self.frame_count
            reactivated_track['lost_frames'] = 0
            reactivated_track['missed_updates'] = 0

            self.active_tracks[track_id] = reactivated_track
            final_matches[det_idx] = track_id

            profile = self.person_profiles.get(track_id)
            if profile:
                self._update_person_profile(profile, detection)

        # 4. Create new tracks
        remaining_unmatched = [i for i, det in unmatched_detections if i not in reid_matches]

        for det_idx in remaining_unmatched:
            detection = detections[det_idx]
            new_id = self._create_new_track(detection)
            final_matches[det_idx] = new_id

        # 5. Handle lost tracks
        self._handle_lost_tracks(final_matches)

        return final_matches

    def _match_with_motion(self, detections: List[Dict]) -> Dict[int, int]:
        """Match detections with tracks using motion prediction"""
        matches = {}

        if not self.active_tracks or not detections:
            return matches

        track_ids = list(self.active_tracks.keys())
        cost_matrix = np.full((len(detections), len(track_ids)), 1.0)

        for det_idx, detection in enumerate(detections):
            for track_idx, track_id in enumerate(track_ids):
                track = self.active_tracks[track_id]

                predicted_bbox, motion_confidence = self._predict_motion(track['kalman'], track['missed_updates'])

                if motion_confidence > self.config.motion_confidence_threshold:
                    iou = self._calculate_iou(detection['bbox'], predicted_bbox)
                    cost_matrix[det_idx, track_idx] = 1.0 - iou
                else:
                    cost_matrix[det_idx, track_idx] = 0.95

        # Hungarian assignment
        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        for det_idx, track_idx in zip(det_indices, track_indices):
            cost = cost_matrix[det_idx, track_idx]
            if cost < (1.0 - self.config.iou_threshold):
                track_id = track_ids[track_idx]
                matches[det_idx] = track_id

        return matches

    def _match_with_appearance(self, unmatched_detections: List[Tuple[int, Dict]]) -> Dict[int, Tuple[int, str]]:
        """Match with active tracks using appearance"""
        matches = {}

        for det_idx, detection in unmatched_detections:
            best_match_id = None
            best_score = 0.0
            best_match_type = ""

            for track_id, track in self.active_tracks.items():
                if track_id in [m for m in matches.values()]:
                    continue

                profile = self.person_profiles.get(track_id)
                if not profile:
                    continue

                similarity, match_type = self._compute_person_similarity(detection, profile)

                if similarity > best_score and similarity > self.config.combined_reid_threshold:
                    best_score = similarity
                    best_match_id = track_id
                    best_match_type = match_type

            if best_match_id:
                matches[det_idx] = (best_match_id, best_match_type)

        return matches

    def _match_with_lost_tracks(self, unmatched_detections: List[Tuple[int, Dict]]) -> Dict[int, Tuple[int, str]]:
        """Re-identification with lost tracks"""
        matches = {}

        for det_idx, detection in unmatched_detections:
            best_match_id = None
            best_score = 0.0
            best_match_type = ""

            for track_id, lost_track in self.lost_tracks.items():
                profile = self.person_profiles.get(track_id)
                if not profile:
                    continue

                similarity, match_type = self._compute_person_similarity(detection, profile)

                reid_threshold = self.config.combined_reid_threshold + 0.1
                if similarity > best_score and similarity > reid_threshold:
                    best_score = similarity
                    best_match_id = track_id
                    best_match_type = match_type

            if best_match_id:
                matches[det_idx] = (best_match_id, best_match_type)

        return matches

    def _compute_person_similarity(self, detection: Dict, profile: Dict) -> Tuple[float, str]:
        """Compute similarity between detection and person profile"""
        similarities = []
        matching_components = []
        weights = []

        # Face similarity
        if detection['face_feature'] is not None and profile.get('face_feature') is not None:
            face_sim = self._compute_feature_similarity(detection['face_feature'], profile['face_feature'])
            if face_sim > self.config.face_reid_threshold:
                similarities.append(face_sim)
                matching_components.append("face")
                weights.append(0.5)

        # Upper body similarity
        if detection['upper_feature'] is not None and profile.get('upper_feature') is not None:
            upper_sim = self._compute_feature_similarity(detection['upper_feature'], profile['upper_feature'])
            if upper_sim > self.config.upper_reid_threshold:
                similarities.append(upper_sim)
                matching_components.append("upper")
                weights.append(0.35)

        # Lower body similarity
        if detection['lower_feature'] is not None and profile.get('lower_feature') is not None:
            lower_sim = self._compute_feature_similarity(detection['lower_feature'], profile['lower_feature'])
            if lower_sim > self.config.lower_reid_threshold:
                similarities.append(lower_sim)
                matching_components.append("lower")
                weights.append(0.15)

        if similarities and weights:
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            combined_sim = sum(s * w for s, w in zip(similarities, normalized_weights))
            match_description = "+".join(matching_components)
            return combined_sim, match_description

        return 0.0, "none"

    def _compute_feature_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between features"""
        if feat1 is None or feat2 is None:
            return 0.0
        try:
            similarity = cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0, 0]
            return max(0.0, similarity)
        except:
            return 0.0

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _create_kalman_filter(self, initial_bbox: np.ndarray) -> KalmanFilter:
        """Create Kalman filter for motion tracking"""
        kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement function
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

        # Initialize state
        x_center = (initial_bbox[0] + initial_bbox[2]) / 2
        y_center = (initial_bbox[1] + initial_bbox[3]) / 2
        width = initial_bbox[2] - initial_bbox[0]
        height = initial_bbox[3] - initial_bbox[1]

        kf.x = np.array([x_center, y_center, width, height, 0, 0, 0, 0])
        kf.P *= 100
        kf.R *= 10
        kf.Q *= 0.1

        return kf

    def _predict_motion(self, kalman_filter: KalmanFilter, missed_updates: int) -> Tuple[np.ndarray, float]:
        """Predict next position using Kalman filter"""
        kalman_filter.predict()

        # Extract bbox from state
        x_center, y_center, width, height = kalman_filter.x[:4]
        predicted_bbox = np.array([
            x_center - width/2,
            y_center - height/2,
            x_center + width/2,
            y_center + height/2
        ])

        # Update confidence based on consecutive misses
        prediction_confidence = max(0.1, 1.0 - (missed_updates * 0.15))

        return predicted_bbox, prediction_confidence

    def _update_kalman_filter(self, kalman_filter: KalmanFilter, measurement_bbox: np.ndarray):
        """Update Kalman filter with new measurement"""
        x_center = (measurement_bbox[0] + measurement_bbox[2]) / 2
        y_center = (measurement_bbox[1] + measurement_bbox[3]) / 2
        width = measurement_bbox[2] - measurement_bbox[0]
        height = measurement_bbox[3] - measurement_bbox[1]

        measurement = np.array([x_center, y_center, width, height])
        kalman_filter.update(measurement)

    def _update_person_profile(self, profile: Dict, detection: Dict):
        """Update person profile with new features"""
        # Update features with exponential moving average
        alpha = 0.3

        if detection['face_feature'] is not None:
            if profile.get('face_feature') is None:
                profile['face_feature'] = detection['face_feature'].copy()
            else:
                profile['face_feature'] = alpha * detection['face_feature'] + (1 - alpha) * profile['face_feature']
                norm = np.linalg.norm(profile['face_feature'])
                if norm > 0:
                    profile['face_feature'] = profile['face_feature'] / norm

        if detection['upper_feature'] is not None:
            if profile.get('upper_feature') is None:
                profile['upper_feature'] = detection['upper_feature'].copy()
            else:
                profile['upper_feature'] = alpha * detection['upper_feature'] + (1 - alpha) * profile['upper_feature']
                norm = np.linalg.norm(profile['upper_feature'])
                if norm > 0:
                    profile['upper_feature'] = profile['upper_feature'] / norm

        if detection['lower_feature'] is not None:
            if profile.get('lower_feature') is None:
                profile['lower_feature'] = detection['lower_feature'].copy()
            else:
                profile['lower_feature'] = alpha * detection['lower_feature'] + (1 - alpha) * profile['lower_feature']
                norm = np.linalg.norm(profile['lower_feature'])
                if norm > 0:
                    profile['lower_feature'] = profile['lower_feature'] / norm

    def _create_new_track(self, detection: Dict) -> int:
        """Create new track and person profile"""
        track = {
            'track_id': self.next_track_id,
            'kalman': self._create_kalman_filter(detection['bbox']),
            'detections': deque([detection], maxlen=100),
            'last_seen': self.frame_count,
            'created_frame': self.frame_count,
            'lost_frames': 0,
            'missed_updates': 0
        }

        profile = {
            'person_id': self.next_track_id,
            'creation_frame': self.frame_count,
            'face_feature': detection['face_feature'].copy() if detection['face_feature'] is not None else None,
            'upper_feature': detection['upper_feature'].copy() if detection['upper_feature'] is not None else None,
            'lower_feature': detection['lower_feature'].copy() if detection['lower_feature'] is not None else None
        }

        self.active_tracks[self.next_track_id] = track
        self.person_profiles[self.next_track_id] = profile

        current_id = self.next_track_id
        self.next_track_id += 1

        return current_id

    def _handle_lost_tracks(self, final_matches: Dict[int, int]):
        """Handle lost tracks and cleanup"""
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if track_id not in final_matches.values():
                track['missed_updates'] += 1
                track['lost_frames'] += 1

                if track['lost_frames'] > self.config.max_lost_frames:
                    if len(track['detections']) >= 10:
                        self.lost_tracks[track_id] = track
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]

        # Cleanup old lost tracks
        tracks_to_cleanup = []
        for track_id, track in self.lost_tracks.items():
            if self.frame_count - track['last_seen'] > self.config.max_lost_frames * 2:
                tracks_to_cleanup.append(track_id)

        for track_id in tracks_to_cleanup:
            del self.lost_tracks[track_id]


# ================== VISUALIZATION MODULE ==================

class VisualizationModule:
    """Handles visualization of tracking results"""

    def __init__(self, config: VisualizationConfig, pose_estimator, visualizer):
        self.config = config
        self.pose_estimator = pose_estimator
        self.visualizer = visualizer

    def draw_tracking_results(self, frame: np.ndarray, pose_results: List, person_assignments: Dict[int, int], active_tracks: Dict) -> np.ndarray:
        """Draw tracking results on frame"""
        if not self.config.enable_visualization or not person_assignments:
            return frame

        # Filter pose results
        filtered_poses = []
        for det_idx, track_id in person_assignments.items():
            if det_idx < len(pose_results):
                filtered_poses.append(pose_results[det_idx])

        if not filtered_poses:
            return frame

        vis_result = frame.copy()

        # Draw poses if enabled
        if self.config.enable_pose_drawing:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            data_samples = merge_data_samples(filtered_poses)
            self.visualizer.add_datasample(
                'result', img_rgb, data_sample=data_samples,
                draw_gt=False, draw_heatmap=False, draw_bbox=False,
                show=False, wait_time=0, out_file=None, kpt_thr=self.config.keypoint_threshold
            )

            vis_result = self.visualizer.get_image()
            vis_result = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)

        # Draw tracking info
        if self.config.enable_bbox_drawing or self.config.enable_id_labels:
            for det_idx, track_id in person_assignments.items():
                if det_idx < len(pose_results):
                    pose = pose_results[det_idx]
                    if hasattr(pose.pred_instances, 'bboxes') and len(pose.pred_instances.bboxes) > 0:
                        bbox = pose.pred_instances.bboxes[0]
                        if torch.is_tensor(bbox):
                            bbox = bbox.cpu().numpy()

                        x1, y1, x2, y2 = bbox[:4].astype(int)

                        # Determine match type and color
                        track = active_tracks.get(track_id)
                        if track:
                            if track['missed_updates'] == 0:
                                if track['created_frame'] == track['last_seen']:
                                    match_type = "New"
                                    color = (0, 255, 255)  # Yellow
                                else:
                                    match_type = "Motion"
                                    color = (0, 255, 0)  # Green
                            else:
                                match_type = "Appearance"
                                color = (255, 0, 0)  # Blue
                        else:
                            match_type = "Re-ID"
                            color = (0, 0, 255)  # Red

                        # Draw bounding box
                        if self.config.enable_bbox_drawing:
                            cv2.rectangle(vis_result, (x1, y1), (x2, y2), color, 2)

                        # Draw ID and match type
                        if self.config.enable_id_labels:
                            text = f"ID {track_id}: {match_type}"
                            cv2.rectangle(vis_result, (x1, y1 - 25), (x1 + len(text) * 12, y1), color, -1)
                            cv2.putText(vis_result, text, (x1 + 2, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return vis_result


# ================== MAIN PIPELINE ORCHESTRATOR ==================

class MultiPersonTrackingPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize modules
        self.detection_module = DetectionModule(config.models, config.processing)
        self.pose_module = PoseEstimationModule(config.models, config.visualization)
        self.feature_module = FeatureExtractionModule(config.features)
        self.tracking_module = TrackingModule(config.processing)
        self.visualization_module = VisualizationModule(
            config.visualization,
            self.pose_module.pose_estimator,
            self.pose_module.visualizer
        ) if config.visualization.enable_visualization else None

        # Performance tracking
        self.timing_stats = defaultdict(list)
        self.global_start_time = None

        # Interruption handling
        self._interrupted = False
        self._proc = None
        self._cap = None

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C"""
        print("\n\nInterrupted! Finalizing video and printing metrics...")
        self._interrupted = True

        # Close ffmpeg stdin to let it finalize the video
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except:
                pass

        # Release video capture
        if self._cap:
            try:
                self._cap.release()
            except:
                pass

        # Wait for ffmpeg to finish
        if self._proc:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        # Print performance stats
        print(f"\nPartial processing complete after {self.tracking_module.frame_count} frames")
        print(f"Total persons tracked: {len(self.tracking_module.person_profiles)}")
        self.print_performance_stats()

        sys.exit(0)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
        """Process single frame"""
        # Detection
        det_start = time.time()
        bboxes = self.detection_module.detect_single(frame)
        self.timing_stats['detection'].append(time.time() - det_start)

        # Pose estimation
        pose_start = time.time()
        pose_results = self.pose_module.estimate_single(frame, bboxes)
        self.timing_stats['pose'].append(time.time() - pose_start)

        # Feature extraction
        feat_start = time.time()
        detections = self.feature_module.extract_features(frame, pose_results, self.tracking_module.frame_count + 1)
        self.timing_stats['features'].append(time.time() - feat_start)

        # Tracking
        track_start = time.time()
        person_assignments = self.tracking_module.update(detections)
        self.timing_stats['tracking'].append(time.time() - track_start)

        # Visualization
        vis_frame = frame
        if self.visualization_module:
            vis_start = time.time()
            vis_frame = self.visualization_module.draw_tracking_results(
                frame, pose_results, person_assignments, self.tracking_module.active_tracks
            )
            self.timing_stats['visualization'].append(time.time() - vis_start)

        return vis_frame, person_assignments

    def process_video(self, input_path: str, output_path: str):
        """Process entire video"""
        self.global_start_time = time.time()

        # Setup signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        # Store references for signal handler
        self._cap = cap

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps):
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup ffmpeg for h264 encoding
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
            "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "18",
            output_path
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # Store reference for signal handler
        self._proc = proc

        try:
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while cap.isOpened() and not self._interrupted:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    try:
                        frame_start = time.time()

                        vis_frame, person_assignments = self.process_frame(frame)

                        # Ensure frame size matches what we told ffmpeg
                        if vis_frame.shape[0] != height or vis_frame.shape[1] != width:
                            vis_frame = cv2.resize(vis_frame, (width, height), interpolation=cv2.INTER_LINEAR)

                        # Write frame to ffmpeg stdin
                        try:
                            proc.stdin.write(vis_frame.tobytes())
                        except BrokenPipeError:
                            # ffmpeg died early; print its error and stop
                            err = proc.stderr.read().decode(errors="ignore")
                            raise RuntimeError(f"ffmpeg exited early.\n{err}")

                        self.timing_stats['total_frame'].append(time.time() - frame_start)

                        # Progress update
                        if self.tracking_module.frame_count % 50 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()
                            print(f"Frame {self.tracking_module.frame_count}: "
                                  f"Active={len(self.tracking_module.active_tracks)}, "
                                  f"Lost={len(self.tracking_module.lost_tracks)}, "
                                  f"Total={len(self.tracking_module.person_profiles)}")

                    except Exception as e:
                        print(f"Error processing frame {self.tracking_module.frame_count}: {e}")
                        # Write original frame on error
                        try:
                            if frame.shape[0] != height or frame.shape[1] != width:
                                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                            proc.stdin.write(frame.tobytes())
                        except BrokenPipeError:
                            break

                    pbar.update(1)

        finally:
            cap.release()
            if proc.stdin:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            rc = proc.wait()
            if rc != 0:
                err = proc.stderr.read().decode(errors="ignore")
                raise RuntimeError(f"ffmpeg failed (code {rc}).\n{err}")

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Processing complete. Output saved: {output_path}")
        print(f"Total persons tracked: {len(self.tracking_module.person_profiles)}")
        self.print_performance_stats()

    def print_performance_stats(self):
        """Print performance statistics"""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)

        if self.timing_stats:
            for operation, times in self.timing_stats.items():
                if times:
                    avg_time = np.mean(times)
                    total_time = np.sum(times)
                    count = len(times)
                    print(f"{operation}:")
                    print(f"  Count: {count}")
                    print(f"  Total: {total_time:.3f}s")
                    print(f"  Average: {avg_time:.3f}s")
                    print(f"  Min: {np.min(times):.3f}s")
                    print(f"  Max: {np.max(times):.3f}s")
                    print()

        if self.global_start_time:
            total_runtime = time.time() - self.global_start_time
            frames_processed = self.tracking_module.frame_count
            fps = frames_processed / total_runtime if total_runtime > 0 else 0
            print(f"Overall Stats:")
            print(f"  Total runtime: {total_runtime:.2f}s")
            print(f"  Frames processed: {frames_processed}")
            print(f"  Processing FPS: {fps:.2f}")

        print("="*60)


# ================== EXAMPLE USAGE ==================

def create_custom_config() -> PipelineConfig:
    """Create a custom configuration"""
    config = PipelineConfig()

    # Example: Disable face features for faster processing
    # config.features.enable_face_features = False

    # Example: Change thresholds
    # config.processing.detection_confidence_threshold = 0.6
    # config.processing.combined_reid_threshold = 0.8

    # Example: Disable visualization
    # config.visualization.enable_visualization = False
    config.visualization.enable_pose_drawing = False

    return config

def main():
    """Main function demonstrating usage"""
    # Create configuration
    config = create_custom_config()

    # Initialize pipeline
    pipeline = MultiPersonTrackingPipeline(config)

    # Process video
    DATA_DIR = "/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external"
    VID_LOCAL_PATH = "/H.L._Home_Videos_AMES_A6Y4Y7X2G1/12-16 month videos/Dec 2018 (14m)/12-29-2018.MOV"
    TARGET_VIDEO_PATH = "/orcd/data/satra/001/users/brukew/test_nb/outputs/modular_pipeline_output.mp4"
    SOURCE_VIDEO_PATH = DATA_DIR + VID_LOCAL_PATH

    pipeline.process_video(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH)

if __name__ == "__main__":
    main()