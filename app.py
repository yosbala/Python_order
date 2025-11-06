import os
import boto3
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from ultralytics import YOLO
import json
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from scipy.spatial.distance import cosine
from skimage.feature import hog
import warnings

warnings.filterwarnings('ignore')

# ---------- AWS CONFIG ----------
AWS_ACCESS_KEY = "AKIATE7TKZGQ4FANLKEC"
AWS_SECRET_KEY = "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"
BUCKET_NAME = "rupos"
REGION = "us-east-1"
FOLDER = "Prod/cakepoint/images/S0/Products/"
# --------------------------------

CACHE_DIR = "cache_images"
INDEX_FILE = "product_features_v3.pkl"
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------- ADVANCED MODEL SETUP ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

if device.type == 'cuda':
    model = model.half()
    torch.backends.cudnn.benchmark = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

yolo_model = YOLO("yolov8n.pt")
yolo_model.fuse()
if device.type == 'cuda':
    yolo_model.to(device)

# ---------- AWS UTILITIES ----------
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=REGION,
    )

def cache_s3_images():
    print("🔄 Checking for new images in S3...")
    s3 = get_s3_client()
    objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FOLDER)
    if "Contents" not in objects:
        print("❌ No images found in S3 folder.")
        return []

    new_images_found = False
    local_paths = []
    for obj in tqdm(objects["Contents"], desc="Syncing", disable=True):
        key = obj["Key"]
        if not key.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        filename = os.path.join(CACHE_DIR, os.path.basename(key))
        last_modified = obj["LastModified"]

        download_file = False
        if not os.path.exists(filename):
            download_file = True
            new_images_found = True
        else:
            local_mtime = os.path.getmtime(filename)
            s3_mtime = last_modified.timestamp()
            if s3_mtime > local_mtime:
                download_file = True
                new_images_found = True

        if download_file:
            s3.download_file(BUCKET_NAME, key, filename)
        local_paths.append(filename)

    if new_images_found and os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)

    return local_paths

# ---------- ROTATION-INVARIANT FEATURES ----------
def compute_rotation_invariant_features(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), visualize=False, feature_vector=True)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hog_features.astype('float32'), hu_moments.astype('float32')

def extract_color_moments(img_cv):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (64, 64))
    moments = []
    for channel in range(3):
        ch = img_rgb[:, :, channel].flatten()
        mean = np.mean(ch)
        std = np.std(ch)
        skew = np.mean(((ch - mean) / (std + 1e-7)) ** 3)
        moments.extend([mean, std, skew])
    return np.array(moments, dtype='float32')

def extract_sift_keypoints(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    sift = cv2.SIFT_create(nfeatures=50)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is not None and len(descriptors) > 0:
        global_descriptor = np.mean(descriptors, axis=0)
        return global_descriptor.astype('float32')
    return np.zeros(128, dtype='float32')

@torch.no_grad()
def extract_deep_features_gpu(images_pil):
    if len(images_pil) == 0:
        return np.array([])
    tensors = torch.stack([transform(img) for img in images_pil]).to(device)
    if device.type == 'cuda':
        tensors = tensors.half()
    features = model(tensors)
    features = features.cpu().float().numpy().reshape(len(images_pil), -1)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-8)
    return features.astype('float32')

def extract_all_features(img_cv, img_pil):
    deep_feat = extract_deep_features_gpu([img_pil])[0]
    hog_feat, hu_feat = compute_rotation_invariant_features(img_cv)
    color_feat = extract_color_moments(img_cv)
    sift_feat = extract_sift_keypoints(img_cv)
    return {
        'deep': deep_feat,
        'hog': hog_feat,
        'hu': hu_feat,
        'color': color_feat,
        'sift': sift_feat
    }

def detect_and_extract_objects(image_path, conf_threshold=0.2):
    img = cv2.imread(image_path)
    if img is None:
        return None, []
    results = yolo_model(image_path, conf=conf_threshold, verbose=False)
    objects_data = []
    for r in results:
        if len(r.boxes) == 0:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w < 20 or h < 20:
                continue
            obj_cv = img[y1:y2, x1:x2]
            if obj_cv.size == 0:
                continue
            obj_pil = Image.fromarray(cv2.cvtColor(obj_cv, cv2.COLOR_BGR2RGB)).convert("RGB")
            features = extract_all_features(obj_cv, obj_pil)
            area = w * h
            aspect_ratio = w / h if h > 0 else 1.0
            objects_data.append({
                'box': (x1, y1, x2, y2),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'features': features
            })
    return img, objects_data

def process_image_for_index(img_path):
    _, objects = detect_and_extract_objects(img_path)
    if objects:
        return (os.path.basename(img_path), objects)
    return None

def build_or_load_index():
    if os.path.exists(INDEX_FILE):
        print("📂 Loading cached features...")
        with open(INDEX_FILE, 'rb') as f:
            return pickle.load(f)
    print("🏗️  Building advanced feature index...")
    data_images = [
        os.path.join(CACHE_DIR, f)
        for f in os.listdir(CACHE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    index = {}
    batch_size = 8
    for i in tqdm(range(0, len(data_images), batch_size), desc="Indexing"):
        batch = data_images[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_image_for_index, batch))
        for result in results:
            if result:
                img_name, objects = result
                index[img_name] = objects
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(index, f)
    print(f"✅ Indexed {len(index)} images")
    return index

# ---------- FAST DEEP FEATURE SHORTLIST ----------
def initial_deep_match(upload_vec, db_index, top_n=25):
    """Shortlist top-N images by cosine similarity (deep features only)."""
    sims = []
    for name, objs in db_index.items():
        all_deep = [o['features']['deep'] for o in objs]
        mean_vec = np.mean(all_deep, axis=0)
        sim = float(np.dot(upload_vec, mean_vec))
        sims.append((name, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [name for name, sim in sims[:top_n]]

# ---------- SIMILARITY PIPELINE ----------
def compute_multi_feature_similarity(feat1, feat2):
    deep_sim = float(np.dot(feat1['deep'], feat2['deep']))
    hog_sim = float(1 - cosine(feat1['hog'], feat2['hog']))
    hu_sim = float(1 - cosine(feat1['hu'], feat2['hu']))
    color_sim = float(1 - cosine(feat1['color'], feat2['color']))
    sift_sim = float(1 - cosine(feat1['sift'], feat2['sift']))
    total_sim = (
        deep_sim * 0.45 +
        hog_sim * 0.20 +
        hu_sim * 0.15 +
        color_sim * 0.15 +
        sift_sim * 0.05
    )
    return total_sim, {
        'deep': deep_sim,
        'hog': hog_sim,
        'hu': hu_sim,
        'color': color_sim,
        'sift': sift_sim
    }

def compare_objects_advanced(obj1, obj2):
    feat1 = obj1['features']
    feat2 = obj2['features']
    similarity, breakdown = compute_multi_feature_similarity(feat1, feat2)
    aspect_diff = abs(obj1['aspect_ratio'] - obj2['aspect_ratio'])
    aspect_penalty = max(0, 1 - aspect_diff / 2)
    final_score = similarity * 0.9 + aspect_penalty * 0.1
    return final_score * 100

def compare_with_db_image(args):
    db_name, db_objs, upload_objs = args
    if not db_objs:
        return None
    scores = []
    for up_obj in upload_objs:
        best_score = 0
        for db_obj in db_objs:
            score = compare_objects_advanced(up_obj, db_obj)
            best_score = max(best_score, score)
        scores.append(best_score)
    if not scores:
        return None
    avg_score = float(np.mean(scores))
    max_score = float(np.max(scores))
    min_score = float(np.min(scores))
    consistency = 100 - (max_score - min_score)
    return {
        "imagename": db_name,
        "accuracy": round(avg_score, 2),
        "max_score": round(max_score, 2),
        "min_score": round(min_score, 2),
        "consistency": round(consistency, 2)
    }

def find_best_match_ultra_fast(upload_path, top_k=5, threshold=88):
    print("🚀 Starting ultra-fast comparison...")
    cache_s3_images()
    db_index = build_or_load_index()
    if not db_index:
        return {"match": False, "error": "No database images indexed"}
    print("📸 Processing upload image...")
    upload_img, upload_objs = detect_and_extract_objects(upload_path)
    if not upload_objs:
        return {"match": False, "error": "No objects detected in upload image"}
    print(f"📦 Detected {len(upload_objs)} objects")

    # 1. Deep-feature-only shortlisting (fast)
    upload_deep = [o['features']['deep'] for o in upload_objs]
    upload_vec = np.mean(upload_deep, axis=0)
    shortlist = initial_deep_match(upload_vec, db_index, top_n=25)
    print(f"📊 Shortlisted {len(shortlist)} candidates for full comparison")

    # 2. Only compare the shortlist
    comparison_args = [
        (db_name, db_index[db_name], upload_objs)
        for db_name in shortlist
    ]
    num_workers = min(mp.cpu_count(), len(shortlist))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        comparison_results = list(tqdm(
            executor.map(compare_with_db_image, comparison_args),
            total=len(comparison_args),
            desc="Matching"
        ))
    results = [r for r in comparison_results if r is not None]
    if not results:
        return {"match": False, "error": "No valid comparisons completed"}
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    best_result = results[0]
    best_result["match"] = bool(best_result.get("accuracy", 0) > threshold)
    vis_img = upload_img.copy()
    color = (0, 255, 0) if best_result["match"] else (0, 0, 255)
    for i, obj_data in enumerate(upload_objs):
        x1, y1, x2, y2 = obj_data['box']
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis_img, f"Obj{i+1}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    accuracy = best_result.get('accuracy', 0)
    status = "✓ MATCH" if best_result["match"] else "✗ NO MATCH"
    cv2.putText(vis_img, f"{status}: {accuracy:.1f}%",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(vis_img, best_result['imagename'][:30],
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out_path = os.path.join("output_visual", "comparison_result.jpg")
    os.makedirs("output_visual", exist_ok=True)
    cv2.imwrite(out_path, vis_img)
    print(f"✅ Result: {best_result['imagename']} ({accuracy:.2f}%)")
    top_matches = []
    for match in results[:top_k]:
        top_matches.append({
            "imagename": str(match["imagename"]),
            "accuracy": float(match["accuracy"]),
            "max_score": float(match["max_score"]),
            "consistency": float(match.get("consistency", 0))
        })
    final_result = {
        "match": bool(best_result["match"]),
        "accuracy": float(best_result["accuracy"]),
        "imagename": str(best_result["imagename"]),
        "max_score": float(best_result.get("max_score", 0)),
        "consistency": float(best_result.get("consistency", 0)),
        "objects_detected": len(upload_objs),
        "top_matches": top_matches
    }
    return final_result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image provided"}))
        exit(1)
    upload_img = sys.argv[1]
    res = find_best_match_ultra_fast(upload_img)
    print(json.dumps(res, indent=2))
