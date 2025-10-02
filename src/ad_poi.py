"""
Person of Interest Image Search System

This script downloads celebrity images, creates vector embeddings, and provides
a Streamlit interface for text-to-image search using CLIP model.

Workflow:
1. Download images from HuggingFace dataset
2. Create vector embeddings for all images
3. Save embeddings and metadata for reuse
4. Launch Streamlit search interface

Instructions:
1. First run (only once) download the images and create the embeddings using: uv run python src/ad_poi.py
2. Then run the Streamlit app using: uv run streamlit run src/ad_poi.py
3. If port 8501 ia not free, then you can run the app using: uv run streamlit run src/ad_poi.py --server.port 8502
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm 
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project directories
data_dir = Path('/home/anuragd/labwork/Person_Of_Interest_Project').expanduser()
img_dir = data_dir / "docs/images/celebA"
img_dir.mkdir(parents=True, exist_ok=True)

# Output files for embeddings and metadata (to avoid re-processing)
# inlieu of src, create a new 'data' directory under the root and possibly ad my prefix to the file names
OUT_VEC = data_dir / "data/index_vectors.npy"
OUT_META = data_dir / "data/index_meta.parquet"

# Dataset configuration
repo_id = "student/celebA"   # HuggingFace dataset ID
split = "train"              # Dataset split to use
max_images = 0               # 0 means all images; set e.g. 500 for quick testing

# Model configuration
MODEL_NAME = "clip-ViT-B-32"  # CLIP model for image-text embeddings

# =============================================================================
# STEP 1: DOWNLOAD IMAGES (only if not already downloaded)
# =============================================================================

def download_images():
    """Download images from HuggingFace dataset if not already present."""
    # Check if images already exist
    existing_images = list(img_dir.glob("*.jpg"))
    if existing_images and max_images == 0:
        print(f"✅ Images already exist ({len(existing_images)} files). Skipping download.")
        return
    
    print(f"Downloading {repo_id}:{split} → {img_dir}")
    
    # Load the dataset from HuggingFace
    ds = load_dataset(repo_id, split=split)
    
    # Save each image as a .jpg file
    for i, row in enumerate(tqdm(ds, desc="Downloading images")):
        img = row["image"]  # Extract image from dataset row
        
        # Convert to PIL Image if needed
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        
        # Save image with consistent naming
        img.save(img_dir / f"{split}_{i:06d}.jpg", "JPEG", quality=90)
        
        # Stop if we've reached the maximum number of images
        if max_images and i + 1 >= max_images:
            break
    
    print(f"✅ Done. Images saved in {img_dir}")

# =============================================================================
# STEP 2: CREATE VECTOR EMBEDDINGS (only if not already created)
# =============================================================================

def create_embeddings():
    """Create vector embeddings for all images if not already created."""
    # Check if embeddings already exist
    if OUT_VEC.exists() and OUT_META.exists():
        print("✅ Embeddings already exist. Skipping vectorization.")
        return
    
    print("Creating vector embeddings...")
    
    # Load the CLIP model for image encoding
    model = SentenceTransformer(MODEL_NAME)
    
    # Process all images and create embeddings
    paths, vecs = [], []
    
    for p in tqdm(sorted(img_dir.glob("**/*")), desc="Processing images"):
        # Only process image files
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            try:
                # Load and convert image to RGB
                img = Image.open(p).convert("RGB")
                
                # Create vector embedding using CLIP model
                v = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
                
                paths.append(str(p))
                vecs.append(v)
                
            except Exception as e:
                print(f"⚠️  Skipping {p}: {e}")
    
    # Stack all vectors into a single numpy array
    vecs = np.vstack(vecs)
    
    # Save embeddings and metadata for future use
    pd.DataFrame({"path": paths}).to_parquet(OUT_META, index=False)
    np.save(OUT_VEC, vecs)
    
    print(f"✅ Indexed {len(paths)} images. Embeddings saved.")

# =============================================================================
# STEP 3: STREAMLIT SEARCH INTERFACE
# =============================================================================

@st.cache_resource
def load_model_and_data():
    """Load model and embeddings with caching to avoid reloading."""
    print("🔄 Loading model and data...")
    try:
        # Load the CLIP model (cached by Streamlit)
        print("📥 Loading CLIP model...")
        model = SentenceTransformer(MODEL_NAME)
        
        # Load pre-computed embeddings and metadata
        if not OUT_META.exists() or not OUT_VEC.exists():
            st.error("❌ Embeddings not found! Please run the script first to create embeddings.")
            st.stop()
        
        print("📥 Loading embeddings and metadata...")
        meta = pd.read_parquet(OUT_META)
        vecs = np.load(OUT_VEC)
        
        print("✅ Model and data loaded successfully!")
        return model, meta, vecs
    
    except Exception as e:
        print(f"❌ Error loading model or data: {e}")
        st.error(f"❌ Error loading model or data: {e}")
        st.stop()

def run_search_app():
    """Launch the Streamlit search interface."""
    # Load model and data (cached)
    model, meta, vecs = load_model_and_data()
    print(f"✅ Model and data loaded")
    # Streamlit UI
    st.title("🔍 Text → Image Search")
    st.markdown("Describe an image and find similar celebrity photos!")
    
    # Display dataset info
    st.info(f"📊 Dataset: {len(meta)} images indexed")
    
    # User input
    q = st.text_input("Describe an image…", placeholder="e.g., 'a person with blonde hair and blue eyes'")
    k = st.slider("Number of results to show", 1, min(12, len(meta)), 6)
    
    # Perform search when user enters a query
    if q:
        with st.spinner("Searching..."):
            try:
                # Encode the text query using CLIP
                qv = model.encode(q, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
                
                # Calculate cosine similarity between query and all image vectors
                sims = cosine_similarity(qv, vecs).ravel()
                
                # Get top-k most similar images
                top = sims.argsort()[::-1][:k]
                
                # Display results in a grid
                cols = st.columns(min(k, 3))
                for i, j in enumerate(top):
                    with cols[i % len(cols)]:
                        # Check if image file exists
                        img_path = meta.loc[j, "path"]
                        if Path(img_path).exists():
                            st.image(img_path, width='stretch')
                            st.caption(f"Similarity: {sims[j]:.3f}")
                        else:
                            st.error(f"Image not found: {img_path}")
                            
            except Exception as e:
                st.error(f"❌ Search error: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Check if running with Streamlit by looking for streamlit in the call stack
import sys
import inspect

def is_running_with_streamlit():
    """Check if the script is being run by Streamlit."""
    for frame_info in inspect.stack():
        if 'streamlit' in frame_info.filename:
            return True
    return False

# If running with Streamlit, run the app directly
if is_running_with_streamlit():
    print("🚀 Running as Streamlit app...")
    run_search_app()

# If running as regular Python script
if __name__ == "__main__":
    print("🔄 Setting up dataset and embeddings...")
    
    # Step 1: Download images (if needed)
    download_images()
    
    # Step 2: Create embeddings (if needed)
    create_embeddings()
    
    print("✅ Setup complete!")
    print("🚀 To run the search app, use: uv run streamlit run src/ad_poi.py")

