import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import radians, cos, sin, asin, sqrt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.stats import pearsonr
from tqdm import tqdm
from itertools import combinations

rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

CSV_PATH = 'camera_locations_cleaned.csv'
VIDEO_DIR = 'tiger_videos_20251009/left'
PNG_DIR = 'tiger_stripe_extraction_output_20260306'
MAP_OUT_DIR = 'tiger_individual_maps'

os.makedirs(MAP_OUT_DIR, exist_ok=True)

def remove_spatial_outliers(df, lat_col='latitude', lon_col='longitude'):
    df_clean = df[(df[lat_col] != 0) & (df[lon_col] != 0) &
                  (df[lat_col].between(-90, 90)) & (df[lon_col].between(-180, 180))].copy()
    for col in [lat_col, lon_col]:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[(df_clean[col] >= Q1 - 1.5 * IQR) & (df_clean[col] <= Q3 + 1.5 * IQR)]
    return df_clean

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371

def task1_map_individual_tigers():
    print(">>> Executing Task 1: Mapping individual tiger distributions and calculating activity centers...")
    df_cameras = pd.read_csv(CSV_PATH)
    df_cameras['camera_id'] = df_cameras['camera_id'].astype(str).str.replace('.0', '', regex=False).str.strip()
    df_clean = remove_spatial_outliers(df_cameras)
    
    tiger_centroids = {}
    
    subdirs = [d for d in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, d))]
    for folder in tqdm(subdirs, desc="Processing tiger maps"):
        if not folder.startswith('left-'): continue
        tiger_id = int(folder.replace('left-', ''))
        
        cam_ids = set()
        for f in os.listdir(os.path.join(VIDEO_DIR, folder)):
            if f.lower().endswith('.mp4'):
                cam_ids.add(f[:6])
                
        df_tiger = df_clean[df_clean['camera_id'].isin(cam_ids)]
        if len(df_tiger) == 0: continue
        
        mean_lat = df_tiger['latitude'].mean()
        mean_lon = df_tiger['longitude'].mean()
        tiger_centroids[tiger_id] = (mean_lat, mean_lon)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(df_clean['longitude'], df_clean['latitude'], c='lightgray', s=10, label='Monitoring Network Stations')
        plt.scatter(df_tiger['longitude'], df_tiger['latitude'], c='red', s=100, marker='*', edgecolor='black', label=f'Tiger {tiger_id} Sightings')
        plt.scatter(mean_lon, mean_lat, c='blue', s=80, marker='X', label='Activity Center')
        
        plt.title(f'Siberian Tiger ID: {tiger_id} Spatial Activity Trajectory', fontsize=14)
        plt.xlabel('Longitude'); plt.ylabel('Latitude')
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(MAP_OUT_DIR, f'tiger_{tiger_id}_map.png'), dpi=200)
        plt.close()
        
    print(f"✅ Task 1 Complete! Calculated coordinates for {len(tiger_centroids)} tigers. Maps saved to {MAP_OUT_DIR}.")
    return tiger_centroids

def task2_calculate_similarities_unbiased():
    print("\n>>> Executing Task 2: Extracting centroid features and calculating unbiased similarity matrix...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    feature_extractor.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tiger_centroids_features = {}
    
    folders = [d for d in os.listdir(PNG_DIR) if os.path.isdir(os.path.join(PNG_DIR, d))]
    for folder in tqdm(folders, desc="Extracting phenotypic centroids"):
        if not folder.isdigit(): continue
        tiger_id = int(folder)
        
        pngs = glob.glob(os.path.join(PNG_DIR, folder, '*.png'))
        if not pngs: continue
        
        feats = []
        for img_path in pngs:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = feature_extractor(tensor).view(-1)
            feats.append(f.cpu().numpy())
            
        mean_feat = np.mean(feats, axis=0)
        mean_feat = mean_feat / np.linalg.norm(mean_feat)
        
        tiger_centroids_features[tiger_id] = mean_feat
        
    similarity_dict = {}
    tiger_ids = sorted(list(tiger_centroids_features.keys()))
    
    for id1, id2 in combinations(tiger_ids, 2):
        feat_A = tiger_centroids_features[id1]
        feat_B = tiger_centroids_features[id2]
        
        sim = np.dot(feat_A, feat_B)
        similarity_dict[(id1, id2)] = sim
        
    print(f"✅ Task 2 Complete! Calculated unbiased similarity for {len(similarity_dict)} pairs.")
    return similarity_dict

def task3_correlation_analysis(centroids, similarities):
    print("\n>>> Executing Task 3: Analyzing correlation between Spatial Distance and Stripe Similarity...")
    
    distances = []
    sim_scores = []
    pair_labels = []
    
    for (id1, id2), sim in similarities.items():
        if id1 in centroids and id2 in centroids:
            loc1, loc2 = centroids[id1], centroids[id2]
            dist = haversine(loc1[1], loc1[0], loc2[1], loc2[0])
            
            distances.append(dist)
            sim_scores.append(sim)
            pair_labels.append(f"{id1}-{id2}")
            
    if len(distances) < 3:
        print("Insufficient pair data for statistical analysis!")
        return
        
    r_val, p_val = pearsonr(distances, sim_scores)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, sim_scores, color='teal', alpha=0.7, edgecolors='white', s=80)
    
    z = np.polyfit(distances, sim_scores, 1)
    p = np.poly1d(z)
    plt.plot(distances, p(distances), "r--", alpha=0.8, linewidth=2, label='Linear Trendline')
    
    plt.title('Siberian Tiger: Territory Distance vs. Stripe Similarity (Ecological Spatial Analysis)', fontsize=15, pad=15)
    plt.xlabel('Activity Center Distance (Kilometers)', fontsize=12)
    plt.ylabel('Max Stripe Cosine Similarity', fontsize=12)
    
    stats_text = f"Pearson r = {r_val:.4f}\nP-value = {p_val:.4f}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
             
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Correlation_Distance_vs_Similarity.png', dpi=300)
    
    print("✅ Task 3 Complete!")
    print("="*40)
    print("📊 Statistical Report Conclusion:")
    print(f"Data Volume: Successfully matched and compared {len(distances)} unique tiger pairs.")
    print(f"Pearson Correlation (r): {r_val:.4f}")
    print(f"Significance (p): {p_val:.4f}")
    
    if p_val < 0.05:
        if r_val < 0:
            print("💡 Ecological Finding: Statistically significant! Greater distance correlates with lower similarity (suggests kin-based spatial diffusion).")
        else:
            print("💡 Ecological Finding: Statistically significant! However, similarity increases with distance (requires further investigation of ecological barriers).")
    else:
        print("💡 Conclusion: No statistically significant linear correlation found (P > 0.05). Stripe similarity does not appear to be determined by territory proximity.")
    print("="*40)
    plt.show()

if __name__ == "__main__":
    tiger_centroids = task1_map_individual_tigers()
    tiger_similarities = task2_calculate_similarities_unbiased()
    task3_correlation_analysis(tiger_centroids, tiger_similarities)
