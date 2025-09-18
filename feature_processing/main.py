import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np
import os

from deepface import DeepFace
from collections import defaultdict, deque
import cv2
import os
from tqdm import tqdm
import torch
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from facenet_pytorch import MTCNN
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import math
import signal
import sys
from collections import defaultdict

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from mmdet.apis import inference_detector, init_detector
from filterpy.kalman import KalmanFilter


# torch.cuda.set_device(0)
device = 'cuda:0'

mmpose_path = os.path.abspath('sailsprep/feature_processing/mmpose')

det_config = mmpose_path + "/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py"
det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
pose_config =  mmpose_path + '/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'


detector = init_detector(det_config, det_checkpoint, device=device)
pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device=device, 
                                    cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=True))))
mtcnn = MTCNN(keep_all=True, device=device, post_process=False, min_face_size=40)

pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

frame_count = 0
next_track_id = 1
active_tracks = {}
lost_tracks = {}
person_profiles = {}

iou_threshold = 0.3
motion_confidence_threshold = 0.5
feature_update_interval = 10
max_lost_frames = 300  # 10 seconds at 30 FPS
face_reid_threshold = 0.75
upper_reid_threshold = 0.65
lower_reid_threshold = 0.6
combined_reid_threshold = 0.7

# TIMING AND PERFORMANCE TRACKING
timing_stats = defaultdict(list)
global_start_time = None

def display_timing():
    print("\n" + "="*60)
    print("PERFORMANCE METRICS (Ctrl+C pressed)")
    print("="*60)

    if timing_stats:
        for operation, times in timing_stats.items():
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

    if global_start_time:
        total_runtime = time.time() - global_start_time
        frames_processed = frame_count
        fps = frames_processed / total_runtime if total_runtime > 0 else 0
        print(f"Overall Stats:")
        print(f"  Total runtime: {total_runtime:.2f}s")
        print(f"  Frames processed: {frames_processed}")
        print(f"  Processing FPS: {fps:.2f}")

    print("="*60)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully and print timing stats"""
    display_timing()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# HELPER FUNCTIONS

def calculate_iou(box1, box2):
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

def compute_feature_similarity(feat1, feat2):
    """Compute cosine similarity between features"""
    if feat1 is None or feat2 is None:
        return 0.0
    try:
        similarity = cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0, 0]
        return max(0.0, similarity)
    except:
        return 0.0

def determine_pose_type(keypoints):
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
        
        # Calculate hip-to-ankle distance (Just made so that I can get upper and lower body features separately in each situation. just wanted to try matching those separately.)
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

def create_kalman_filter(initial_bbox):
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

def predict_motion(kalman_filter, missed_updates):
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

def update_kalman_filter(kalman_filter, measurement_bbox):
    """Update Kalman filter with new measurement"""
    x_center = (measurement_bbox[0] + measurement_bbox[2]) / 2
    y_center = (measurement_bbox[1] + measurement_bbox[3]) / 2
    width = measurement_bbox[2] - measurement_bbox[0]
    height = measurement_bbox[3] - measurement_bbox[1]
    
    measurement = np.array([x_center, y_center, width, height])
    kalman_filter.update(measurement)

# FEATURE EXTRACTION FUNCTIONS


def extract_face_region(frame, keypoints, bbox):
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

def extract_upper_body_region(frame, keypoints, bbox, pose_type):
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

def extract_lower_body_region(frame, keypoints, bbox, pose_type):
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

def compute_lbp_histogram(gray_image):
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

def extract_face_feature(face_roi):
    """Extract face feature using DeepFace"""
    try:
        # Validate with MTCNN
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(face_rgb)
        
        if boxes is None or probs is None or len(boxes) == 0:
            return None
        
        best_prob = float(np.max(probs))
        if best_prob < 0.75:
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
        else:
            return None
        
        return embedding, best_prob
    except:
        return None

def extract_body_feature(roi):
    """Extract appearance feature for body region"""
    try:
        if roi.shape[0] < 30 or roi.shape[1] < 20:
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
        lbp_hist = compute_lbp_histogram(gray_resized)
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
            combined_feature = combined_feature / norm
        else:
            return None
        
        return combined_feature
    except:
        return None


# TRACKING FUNCTIONS

def create_detection(frame, pose):
    """Create detection from pose result"""
    if not hasattr(pose.pred_instances, 'bboxes') or len(pose.pred_instances.bboxes) == 0:
        return None
    
    bbox = pose.pred_instances.bboxes[0]
    if hasattr(bbox, 'cpu'):
        bbox = bbox.cpu().numpy()
    
    if len(bbox) < 4 or bbox[2] - bbox[0] < 50 or bbox[3] - bbox[1] < 100:
        return None
    
    keypoints = pose.pred_instances.keypoints[0]
    confidence = float(bbox[4]) if len(bbox) > 4 else 1.0
    pose_type = determine_pose_type(keypoints)
    
    # Extract features
    face_feature = None
    upper_feature = None
    lower_feature = None
    
    if frame_count % feature_update_interval == 0:
        # Face feature
        face_result = extract_face_region(frame, keypoints, bbox[:4])
        if face_result:
            face_roi, _ = face_result
            face_feat = extract_face_feature(face_roi)
            if face_feat:
                face_feature, _ = face_feat
        
        # Lower body feature
        lower_result = extract_lower_body_region(frame, keypoints, bbox[:4], pose_type)
        if lower_result:
            lower_roi, _ = lower_result
            lower_feature = extract_body_feature(lower_roi)
    
    # Always extract upper body
    upper_result = extract_upper_body_region(frame, keypoints, bbox[:4], pose_type)
    if upper_result:
        upper_roi, _ = upper_result
        upper_feature = extract_body_feature(upper_roi)
    
    return {
        'bbox': bbox[:4],
        'keypoints': keypoints,
        'confidence': confidence,
        'pose_type': pose_type,
        'face_feature': face_feature,
        'upper_feature': upper_feature,
        'lower_feature': lower_feature
    }

def match_with_motion(detections):
    """Match detections with tracks using motion prediction"""
    matches = {}
    
    if not active_tracks or not detections:
        return matches
    
    track_ids = list(active_tracks.keys())
    cost_matrix = np.full((len(detections), len(track_ids)), 1.0)
    
    for det_idx, detection in enumerate(detections):
        for track_idx, track_id in enumerate(track_ids):
            track = active_tracks[track_id]
            
            predicted_bbox, motion_confidence = predict_motion(track['kalman'], track['missed_updates'])
            
            if motion_confidence > motion_confidence_threshold:
                iou = calculate_iou(detection['bbox'], predicted_bbox)
                cost_matrix[det_idx, track_idx] = 1.0 - iou
            else:
                cost_matrix[det_idx, track_idx] = 0.95
    
    # Hungarian assignment
    det_indices, track_indices = linear_sum_assignment(cost_matrix)
    
    for det_idx, track_idx in zip(det_indices, track_indices):
        cost = cost_matrix[det_idx, track_idx]
        if cost < (1.0 - iou_threshold):
            track_id = track_ids[track_idx]
            matches[det_idx] = track_id
    
    return matches

def compute_person_similarity(detection, profile):
    """Compute similarity between detection and person profile"""
    similarities = []
    matching_components = []
    weights = []
    
    # Face similarity
    if detection['face_feature'] is not None and profile.get('face_feature') is not None:
        face_sim = compute_feature_similarity(detection['face_feature'], profile['face_feature'])
        if face_sim > face_reid_threshold:
            similarities.append(face_sim)
            matching_components.append("face")
            weights.append(0.5)
    
    # Upper body similarity
    if detection['upper_feature'] is not None and profile.get('upper_feature') is not None:
        upper_sim = compute_feature_similarity(detection['upper_feature'], profile['upper_feature'])
        if upper_sim > upper_reid_threshold:
            similarities.append(upper_sim)
            matching_components.append("upper")
            weights.append(0.35)
    
    # Lower body similarity
    if detection['lower_feature'] is not None and profile.get('lower_feature') is not None:
        lower_sim = compute_feature_similarity(detection['lower_feature'], profile['lower_feature'])
        if lower_sim > lower_reid_threshold:
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

def match_with_appearance(unmatched_detections):
    """Match with active tracks using appearance"""
    matches = {}
    
    for det_idx, detection in unmatched_detections:
        best_match_id = None
        best_score = 0.0
        best_match_type = ""
        
        for track_id, track in active_tracks.items():
            if track_id in matches.values():
                continue
            
            profile = person_profiles.get(track_id)
            if not profile:
                continue
            
            similarity, match_type = compute_person_similarity(detection, profile)
            
            if similarity > best_score and similarity > combined_reid_threshold:
                best_score = similarity
                best_match_id = track_id
                best_match_type = match_type
        
        if best_match_id:
            matches[det_idx] = (best_match_id, best_match_type)
    
    return matches

def match_with_lost_tracks(unmatched_detections):
    """Re-identification with lost tracks"""
    matches = {}
    
    for det_idx, detection in unmatched_detections:
        best_match_id = None
        best_score = 0.0
        best_match_type = ""
        
        for track_id, lost_track in lost_tracks.items():
            profile = person_profiles.get(track_id)
            if not profile:
                continue
            
            similarity, match_type = compute_person_similarity(detection, profile)
            
            reid_threshold = combined_reid_threshold + 0.1
            if similarity > best_score and similarity > reid_threshold:
                best_score = similarity
                best_match_id = track_id
                best_match_type = match_type
        
        if best_match_id:
            matches[det_idx] = (best_match_id, best_match_type)
    
    return matches

def update_person_profile(profile, detection):
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

def create_new_track(detection):
    """Create new track and person profile"""
    global next_track_id
    
    track = {
        'track_id': next_track_id,
        'kalman': create_kalman_filter(detection['bbox']),
        'detections': deque([detection], maxlen=100),
        'last_seen': frame_count,
        'created_frame': frame_count,
        'lost_frames': 0,
        'missed_updates': 0
    }
    
    profile = {
        'person_id': next_track_id,
        'creation_frame': frame_count,
        'face_feature': detection['face_feature'].copy() if detection['face_feature'] is not None else None,
        'upper_feature': detection['upper_feature'].copy() if detection['upper_feature'] is not None else None,
        'lower_feature': detection['lower_feature'].copy() if detection['lower_feature'] is not None else None
    }
    
    active_tracks[next_track_id] = track
    person_profiles[next_track_id] = profile
    
    current_id = next_track_id
    next_track_id += 1
    
    return current_id

def update_tracking_system(detections):
    """Main tracking update function"""
    global frame_count
    frame_count += 1
    
    # 1. Motion-based matching
    motion_matches = match_with_motion(detections)
    final_matches = {}
    
    # Update matched tracks
    for det_idx, track_id in motion_matches.items():
        detection = detections[det_idx]
        track = active_tracks[track_id]
        
        # Update Kalman filter
        update_kalman_filter(track['kalman'], detection['bbox'])
        
        # Update track
        track['detections'].append(detection)
        track['last_seen'] = frame_count
        track['lost_frames'] = 0
        track['missed_updates'] = 0
        
        # Update profile
        if frame_count % feature_update_interval == 0:
            profile = person_profiles.get(track_id)
            if profile:
                update_person_profile(profile, detection)
        
        final_matches[det_idx] = track_id
    
    # 2. Appearance-based matching
    unmatched_detections = [(i, det) for i, det in enumerate(detections) if i not in motion_matches]
    appearance_matches = match_with_appearance(unmatched_detections)
    
    for det_idx, (track_id, match_type) in appearance_matches.items():
        detection = detections[det_idx]
        track = active_tracks[track_id]
        
        update_kalman_filter(track['kalman'], detection['bbox'])
        track['detections'].append(detection)
        track['last_seen'] = frame_count
        track['lost_frames'] = 0
        track['missed_updates'] = 0
        
        if frame_count % feature_update_interval == 0:
            profile = person_profiles.get(track_id)
            if profile:
                update_person_profile(profile, detection)
        
        final_matches[det_idx] = track_id
        print(f"ID {track_id}: Appearance ({match_type})")
    
    # Update unmatched list
    unmatched_detections = [(i, det) for i, det in unmatched_detections if i not in appearance_matches]
    
    # 3. Re-identification with lost tracks
    reid_matches = match_with_lost_tracks(unmatched_detections)
    
    for det_idx, (track_id, match_type) in reid_matches.items():
        detection = detections[det_idx]
        
        # Reactivate lost track
        reactivated_track = lost_tracks.pop(track_id)
        reactivated_track['kalman'] = create_kalman_filter(detection['bbox'])
        reactivated_track['detections'].append(detection)
        reactivated_track['last_seen'] = frame_count
        reactivated_track['lost_frames'] = 0
        reactivated_track['missed_updates'] = 0
        
        active_tracks[track_id] = reactivated_track
        final_matches[det_idx] = track_id
        
        profile = person_profiles.get(track_id)
        if profile:
            update_person_profile(profile, detection)
        
        print(f"ID {track_id}: Re-identified ({match_type})")
    
    # 4. Create new tracks
    remaining_unmatched = [i for i, det in unmatched_detections if i not in reid_matches]
    
    for det_idx in remaining_unmatched:
        detection = detections[det_idx]
        new_id = create_new_track(detection)
        final_matches[det_idx] = new_id
        print(f"ID {new_id}: New")
    
    # 5. Handle lost tracks
    tracks_to_remove = []
    for track_id, track in active_tracks.items():
        if track_id not in final_matches.values():
            track['missed_updates'] += 1
            track['lost_frames'] += 1
            
            if track['lost_frames'] > max_lost_frames:
                if len(track['detections']) >= 10:
                    lost_tracks[track_id] = track
                tracks_to_remove.append(track_id)
    
    for track_id in tracks_to_remove:
        if track_id in active_tracks:
            del active_tracks[track_id]
    
    # Cleanup old lost tracks
    tracks_to_cleanup = []
    for track_id, track in lost_tracks.items():
        if frame_count - track['last_seen'] > max_lost_frames * 2:
            tracks_to_cleanup.append(track_id)
    
    for track_id in tracks_to_cleanup:
        del lost_tracks[track_id]
    
    return final_matches


# VISUALIZATION FUNCTIONS

def draw_simple_tracking(frame, pose_results, person_assignments):
    """Simple visualization with just ID and match type"""
    if not person_assignments:
        return frame
    
    # Filter pose results
    filtered_poses = []
    for det_idx, track_id in person_assignments.items():
        if det_idx < len(pose_results):
            filtered_poses.append(pose_results[det_idx])
    
    if not filtered_poses:
        return frame
    
    # Draw poses
    # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # data_samples = merge_data_samples(filtered_poses)
    # visualizer.add_datasample(
    #     'result', img_rgb, data_sample=data_samples,
    #     draw_gt=False, draw_heatmap=False, draw_bbox=False,
    #     show=False, wait_time=0, out_file=None, kpt_thr=0.3
    # )
    
    # vis_result = visualizer.get_image()
    # vis_result = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)

    vis_result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Draw tracking info
    for det_idx, track_id in person_assignments.items():
        if det_idx < len(pose_results):
            pose = pose_results[det_idx]
            if hasattr(pose.pred_instances, 'bboxes') and len(pose.pred_instances.bboxes) > 0:
                bbox = pose.pred_instances.bboxes[0]
                if torch.is_tensor(bbox):
                    bbox = bbox.cpu().numpy()
                
                x1, y1, x2, y2 = bbox[:4].astype(int)
                
                # Determine match type
                track = active_tracks.get(track_id)
                if track:
                    if track['missed_updates'] == 0:
                        if track['created_frame'] == frame_count:
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
                cv2.rectangle(vis_result, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID and match type
                text = f"ID {track_id}: {match_type}"
                cv2.rectangle(vis_result, (x1, y1 - 25), (x1 + len(text) * 8, y1), color, -1)
                cv2.putText(vis_result, text, (x1 + 2, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_result

# MAIN 

def process_video(input_path, output_path):
    """Process video with tracking"""
    global frame_count, next_track_id, active_tracks, lost_tracks, person_profiles, global_start_time

    # Reset global variables
    frame_count = 0
    next_track_id = 1
    active_tracks = {}
    lost_tracks = {}
    person_profiles = {}
    global_start_time = time.time()
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print("Error: Could not create video writer")
        cap.release()
        return
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                frame_start_time = time.time()

                # Person detection
                scope = detector.cfg.get('default_scope', 'mmdet')
                if scope is not None:
                    init_default_scope(scope)

                det_start = time.time()
                detect_result = inference_detector(detector, frame)
                det_time = time.time() - det_start
                timing_stats['person_detection'].append(det_time)

                # Process detection results
                process_start = time.time()
                pred_instance = detect_result.pred_instances.cpu().numpy()
                bboxes = np.concatenate(
                    (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
                bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                            pred_instance.scores > 0.5)]
                bboxes = bboxes[nms(bboxes, 0.7)][:, :4]
                process_time = time.time() - process_start
                timing_stats['bbox_processing'].append(process_time)

                # Pose estimation
                pose_start = time.time()
                pose_results = inference_topdown(pose_estimator, frame, bboxes)
                pose_time = time.time() - pose_start
                timing_stats['pose_estimation'].append(pose_time)

                # Create detections (includes feature extraction)
                feature_start = time.time()
                detections = []
                for pose in pose_results:
                    detection = create_detection(frame, pose)
                    if detection:
                        detections.append(detection)
                feature_time = time.time() - feature_start
                timing_stats['feature_extraction'].append(feature_time)

                # Update tracking
                track_start = time.time()
                person_assignments = update_tracking_system(detections)
                track_time = time.time() - track_start
                timing_stats['tracking_update'].append(track_time)

                # Visualization
                vis_start = time.time()
                vis_frame = draw_simple_tracking(frame, pose_results, person_assignments)
                writer.write(vis_frame)
                vis_time = time.time() - vis_start
                timing_stats['visualization'].append(vis_time)

                # writer.write(frame)

                # Total frame time
                total_frame_time = time.time() - frame_start_time
                timing_stats['total_frame'].append(total_frame_time)
                
                # Progress update every 50 frames
                if frame_count % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"Frame {frame_count}: Active={len(active_tracks)}, Lost={len(lost_tracks)}, Total={len(person_profiles)}")
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                writer.write(frame)
            
            pbar.update(1)
    

    cap.release()
    writer.release()
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Processing complete. Output saved: {output_path}")
    print(f"Total persons tracked: {len(person_profiles)}")

def process(input_folder, output_folder):

    
    os.makedirs(output_folder, exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend([f for f in os.listdir(input_folder) if f.lower().endswith(ext)])
    
    for i, video_file in enumerate(video_files):
        print(f"\nProcessing video {i+1}/{len(video_files)}: {video_file}")
        
        input_path = os.path.join(input_folder, video_file)
        output_filename = os.path.splitext(video_file)[0] + '_tracked.mp4'
        output_path = os.path.join(output_folder, output_filename)
        process_video(input_path, output_path)    
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n outputs saved to  {output_folder}")
    display_timing()

if __name__ == "__main__":
    DATA_DIR = "/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external"
    VID_LOCAL_PATH = "/H.L._Home_Videos_AMES_A6Y4Y7X2G1/12-16 month videos/Dec 2018 (14m)/12-29-2018.MOV"
    TARGET_VIDEO_PATH = "/orcd/data/satra/001/users/brukew/test_nb/outputs/aparna_testing_h100.mp4"
    # OUT_JSON_PATH = "detections_rfdetr.json"
    SOURCE_VIDEO_PATH = DATA_DIR + VID_LOCAL_PATH
    process_video(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH)
    pass