import cv2
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS 
from collections import namedtuple
import base64
import traceback 

# --- Configuration for High Accuracy & Speed ---
DEFAULT_FOLDER_PATH = './assets' 
SIFT_RATIO_THRESHOLD = 0.70 
MIN_GOOD_MATCHES = 150 
MIN_HOMOGRAPHY_INLIERS = 25 
RANSAC_REPROJECTION_THRESHOLD = 1.0


# Named tuple to store pre-processed reference image data
ReferenceImage = namedtuple('ReferenceImage', ['filename', 'gray_kp', 'gray_des', 'width', 'height'])

# Global storage for pre-calculated SIFT data
REFERENCE_IMAGES_DATA = []

# --- SIFT Initialization ---
try:
    SIFT_DETECTOR = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) 
    BF_MATCHER = cv2.FlannBasedMatcher(index_params, search_params)
    print("SIFT and FLANN Matcher Initialized.")
except Exception as e:
    print(f"Matcher initialization error: {e}. Falling back to BFMatcher (less efficient).")
    BF_MATCHER = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


# --- Core Initialization and Matching Logic (Same as before) ---

def load_reference_images_from_folder(folder_path):
    global REFERENCE_IMAGES_DATA
    REFERENCE_IMAGES_DATA.clear()
    
    if not os.path.exists(folder_path):
        print(f"FATAL ERROR: Reference image folder not found at {folder_path}. Detection will fail.")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        
        color_img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        if color_img is None: continue
        
        height, width, _ = color_img.shape
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        kp, des = SIFT_DETECTOR.detectAndCompute(gray_img, None)
        
        if des is not None and des.dtype != np.float32:
            des = des.astype(np.float32)

        if des is None or len(kp) < 20: 
            print(f"Warning: Not enough keypoints in {filename}. Skipping.")
            continue

        REFERENCE_IMAGES_DATA.append(ReferenceImage(filename, kp, des, width, height))
    
    print(f"Successfully loaded {len(REFERENCE_IMAGES_DATA)} reference images.")


def find_best_match(live_frame_gray):
    if not REFERENCE_IMAGES_DATA:
        return None, 0, None 

    kp_frame, des_frame = SIFT_DETECTOR.detectAndCompute(live_frame_gray, None)
    
    if des_frame is not None and des_frame.dtype != np.float32:
        des_frame = des_frame.astype(np.float32)

    if des_frame is None or len(kp_frame) < 2:
        return None, 0, None

    best_match_count = 0 
    best_matched_filename = None
    best_corners = None 

    for ref_img_data in REFERENCE_IMAGES_DATA:
        if ref_img_data.gray_des is None: continue

        # --- 1. SIFT Match Check ---
        try:
            raw_matches = BF_MATCHER.knnMatch(ref_img_data.gray_des, des_frame, k=2)
        except cv2.error:
            continue
        
        good_sift_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < SIFT_RATIO_THRESHOLD * n.distance:
                    good_sift_matches.append(m)

        current_sift_count = len(good_sift_matches)

        if current_sift_count >= MIN_GOOD_MATCHES:
            
            # --- 2. Homography Check (Geometric Stability) ---
            src_pts = np.float32([ref_img_data.gray_kp[m.queryIdx].pt for m in good_sift_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_sift_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJECTION_THRESHOLD) 

            if H is not None:
                inlier_count = np.sum(mask)

                if inlier_count >= MIN_HOMOGRAPHY_INLIERS:
                    
                    # --- 3. Calculate Bounding Box Corners ---
                    h, w = ref_img_data.height, ref_img_data.width
                    corners_ref = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    
                    corners_live = cv2.perspectiveTransform(corners_ref, H)
                    
                    if inlier_count > best_match_count: 
                        best_match_count = inlier_count
                        best_matched_filename = ref_img_data.filename
                        best_corners = corners_live.reshape(-1).tolist() 


    if best_matched_filename and best_corners:
        return best_matched_filename, best_match_count, best_corners
    else:
        return None, best_match_count, None

# --- Flask App Setup ---

app = Flask(__name__)
CORS(app) 

with app.app_context():
    load_reference_images_from_folder(DEFAULT_FOLDER_PATH)

@app.route('/match', methods=['POST'])
def match_image():
    data = request.json
    
    if 'image_data' not in data:
        return jsonify({"error": "Missing image_data"}), 400

    try:
        image_b64 = data['image_data'].split(',')[1] 
        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        live_frame_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if live_frame_gray is None:
            return jsonify({"error": "Could not decode image"}), 400

        matched_filename, match_count, box_coords = find_best_match(live_frame_gray)

        if matched_filename:
            return jsonify({
                "match": matched_filename,
                "confidence": match_count,
                "box_coords": box_coords
            })
        else:
            return jsonify({
                "match": None,
                "confidence": match_count,
                "box_coords": None
            })

    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc() 
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)