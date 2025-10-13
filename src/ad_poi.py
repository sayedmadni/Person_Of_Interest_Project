"""
🎭 Celebrity Search AI - Person of Interest Image Search System

A Streamlit application that allows users to search for celebrities using natural language
descriptions. The system uses CLIP (Contrastive Language-Image Pre-training) model to
find the most similar celebrity images based on text queries.

Features:
- AI-powered text-to-image search using CLIP model
- Beautiful, soothing UI with custom styling
- Celebrity image gallery with toggle functionality
- Query validation and safety features
- Real-time similarity scoring with progress bars

Workflow:
1. Download celebrity images from HuggingFace dataset
2. Create vector embeddings for all images using CLIP
3. Save embeddings and metadata for reuse
4. Launch Streamlit search interface

Usage:
1. First run: uv run python src/ad_poi.py (downloads images and creates embeddings)
2. Run app: uv run streamlit run src/ad_poi.py
3. Alternative port: uv run streamlit run src/ad_poi.py --server.port 8502

Author: AI Assistant
Date: 2024
"""

# Standard library imports
import os
import glob
import base64
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Project directories and file paths
data_dir = Path('/home/sayedf/LLM_Bootcamp/Person_Of_Interest_Project').expanduser()
img_dir = data_dir / "docs/images/celebA"  # Directory for celebrity images
img_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

# Output files for embeddings and metadata (cached to avoid re-processing)
OUT_VEC = data_dir / "data/index_vectors.npy"      # Vector embeddings file
OUT_META = data_dir / "data/index_meta.parquet"    # Image metadata file

# Dataset configuration
repo_id = "student/celebA"   # HuggingFace dataset identifier
split = "train"              # Dataset split to download
max_images = 0               # Maximum images to process (0 = all images, set to 500 for quick testing)

# AI Model configuration
MODEL_NAME = "clip-ViT-B-32"  # CLIP model variant for image-text embeddings

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
    
    # Custom CSS styling
    st.markdown("""
    <style>
    /* Remove default Streamlit spacing */
    .main {
        padding-top: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    /* Main container styling with borders - maximize space usage */
    .main .block-container {
        padding-top: 0;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
        margin: 0;
        border: 3px solid #7FB3D3;
        border-radius: 25px;
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        box-shadow: 0 20px 40px rgba(127, 179, 211, 0.1);
        animation: containerGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes containerGlow {
        from { box-shadow: 0 20px 40px rgba(127, 179, 211, 0.1); }
        to { box-shadow: 0 25px 50px rgba(127, 179, 211, 0.2); }
    }
    
    /* Title styling with animated border - positioned at top, minimal spacing */
    .stApp h1 {
        color: #2c5aa0;
        font-size: 3.2rem;
        font-weight: 700;
        text-align: center;
        margin-top: 0;
        margin-bottom: 0.5rem;
        border: 4px solid #7FB3D3;
        border-radius: 20px;
        padding: 0.5rem;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(240, 248, 255, 0.95));
        box-shadow: 0 8px 32px rgba(127, 179, 211, 0.3);
        backdrop-filter: blur(10px);
        position: relative;
        z-index: 10;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        letter-spacing: 1px;
    }
    
    @keyframes titleShimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Header styling with animated borders - minimal spacing */
    .stApp h2 {
        color: #2C3E50;
        border: 2px solid #7FB3D3;
        border-radius: 15px;
        padding: 0.5rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, rgba(127, 179, 211, 0.1), rgba(152, 216, 200, 0.1));
        box-shadow: 0 8px 25px rgba(127, 179, 211, 0.15);
        animation: headerPulse 2s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .stApp h2::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        animation: headerShine 3s infinite;
    }
    
    @keyframes headerPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    @keyframes headerShine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Input styling with enhanced borders */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 3px solid #B8D4E3;
        padding: 0.75rem 1rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #ffffff, #f0f8ff);
        box-shadow: inset 0 2px 10px rgba(127, 179, 211, 0.1);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #7FB3D3;
        box-shadow: 0 0 0 3px rgba(127, 179, 211, 0.2), inset 0 2px 10px rgba(0, 0, 0, 0.05);
        animation: inputFocus 0.3s ease;
    }
    
    @keyframes inputFocus {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Selectbox styling with soothing colors */
    .stSelectbox > div > div > select {
        border-radius: 20px;
        border: 3px solid #B8D4E3;
        padding: 0.75rem 1rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #ffffff, #f0f8ff);
        box-shadow: inset 0 2px 10px rgba(127, 179, 211, 0.1);
        color: #2C3E50;
    }
    
    .stSelectbox > div > div > select:focus {
        border-color: #7FB3D3;
        box-shadow: 0 0 0 3px rgba(127, 179, 211, 0.2), inset 0 2px 10px rgba(0, 0, 0, 0.05);
        outline: none;
    }
    
    .stSelectbox > div > div > select:hover {
        border-color: #7FB3D3;
        box-shadow: 0 4px 15px rgba(127, 179, 211, 0.2);
    }
    
    /* Button styling with animated borders */
    .stButton > button {
        background: linear-gradient(45deg, #7FB3D3, #98D8C8);
        color: white;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(127, 179, 211, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(127, 179, 211, 0.4);
        border-color: rgba(255, 255, 255, 0.6);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Toggle styling with borders */
    .stToggle > label > div {
        background-color: #7FB3D3;
        border: 2px solid #98D8C8;
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    
    .stToggle > label > div:hover {
        box-shadow: 0 4px 15px rgba(127, 179, 211, 0.3);
        transform: scale(1.05);
    }
    
    /* Image container styling with animated borders */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 3px solid transparent;
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(45deg, #7FB3D3, #98D8C8) border-box;
    }
    
    .stImage:hover {
        transform: scale(1.05) rotate(1deg);
        box-shadow: 0 15px 35px rgba(127, 179, 211, 0.2);
    }
    
    /* Info box styling with animated borders */
    .stAlert {
        border-radius: 15px;
        border: 3px solid #7FB3D3;
        box-shadow: 0 8px 25px rgba(127, 179, 211, 0.15);
        animation: alertPulse 2s ease-in-out infinite;
        background: linear-gradient(135deg, rgba(127, 179, 211, 0.05), rgba(152, 216, 200, 0.05));
    }
    
    @keyframes alertPulse {
        0%, 100% { border-color: #7FB3D3; }
        50% { border-color: #98D8C8; }
    }
    
    /* Sidebar styling with borders */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-right: 4px solid #FF6B6B;
        box-shadow: 5px 0 20px rgba(255, 107, 107, 0.1);
    }
    
    /* Metric styling with enhanced borders - minimal spacing, no animations */
    .stMetric {
        background: linear-gradient(135deg, #ffffff, #f0f8ff);
        padding: 0.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(127, 179, 211, 0.2);
        border: 3px solid #7FB3D3;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        margin: 0.2rem;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(127, 179, 211, 0.2);
        border-color: #98D8C8;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7FB3D3, #98D8C8);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(127, 179, 211, 0.3);
        animation: progressGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes progressGlow {
        from { box-shadow: 0 2px 10px rgba(127, 179, 211, 0.3); }
        to { box-shadow: 0 4px 20px rgba(127, 179, 211, 0.5); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Streamlit UI - Title and subtitle in one box
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%); 
                border: 3px solid #7FB3D3; border-radius: 25px; 
                padding: 1.5rem; margin-bottom: 1rem; 
                box-shadow: 0 20px 40px rgba(127, 179, 211, 0.1);">
        <div style="text-align: center;">
            <h1 style="margin: 0 0 0.5rem 0; background: linear-gradient(90deg, #7FB3D3, #98D8C8, #7FB3D3); 
                       background-size: 200% 200%; -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent; background-clip: text; 
                       font-size: 3rem; font-weight: 700; animation: titleShimmer 2s ease-in-out infinite;">
                🎭 Celebrity Search AI
            </h1>
            <h3 style="margin: 0; color: #2C3E50; font-style: italic; font-size: 1.2rem;">
                *Discover celebrities through AI-powered image search*
            </h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Gallery buttons - View and Close options side by side
    col2, col3, col4 = st.columns([1, 1, 1])
    
    # View Gallery button
    with col2:
        try:
            image_files = glob.glob(os.path.join(img_dir, "*.jpg"))
            
            if st.button("🖼️ View Gallery", help=f"Click to view {len(image_files)} celebrity images"):
                st.session_state.show_gallery = True
                st.rerun()
                
        except Exception as e:
            st.error(f"Error loading images: {e}")
    
    # Close Gallery button (only show when gallery is open)
    with col4:
        if st.session_state.get('show_gallery', False):
            if st.button("❌ Close Gallery", help="Click to hide the gallery"):
                st.session_state.show_gallery = False
                st.rerun()
    
    # Show gallery if button was clicked
    if st.session_state.get('show_gallery', False):
        st.markdown("---")
        st.markdown("## 🖼️ Celebrity Gallery")
        
        with st.spinner("Loading gallery..."):
            try:
                image_files = glob.glob(os.path.join(img_dir, "*.jpg"))
                if image_files:
                    st.markdown(f"**📸 {len(image_files)} celebrity images:**")
                    
                    # Create a scrollable container
                    with st.container():
                        # Display all images in rows of 3
                        for i in range(0, len(image_files), 3):
                            # Create columns for this row
                            row_cols = st.columns(3)
                            
                            for j in range(3):
                                if i + j < len(image_files):
                                    image_file = image_files[i + j]
                                    caption = os.path.basename(image_file)
                                    with row_cols[j]:
                                        st.image(image_file, caption=caption, width=180)
                else:
                    st.write("No images found in the gallery directory.")
            except Exception as e:
                st.error(f"❌ Error loading gallery: {e}")
    

    # Add some visual flair
    st.markdown("---")
    
    # Add a metrics section for better visual appeal - spread horizontally
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Total Images", f"{len(meta):,}")
    with col2:
        st.metric("🔍 Search Model", "CLIP ViT-B-32")
    
    st.markdown("---")
    
    # Search section with enhanced styling
    st.markdown("## 🔍 AI-Powered Search")
    st.markdown("**Describe what you're looking for and let AI find the perfect celebrity match!**")
    
    # Initialize session state for search query
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    # User input with better layout - spread horizontally
    col1, col2 = st.columns([5, 2])
    
    with col1:
        q = st.text_input(
            "🎯 Search Query", 
            value=st.session_state.search_query,
            placeholder="e.g., 'a person with blonde hair and blue eyes smiling'",
            help="Describe the celebrity you're looking for in natural language",
            key="search_input"
        )
        
        # Update session state with current input
        st.session_state.search_query = q
    
    with col2:
        k = st.selectbox(
            "📊 Results", 
            options=list(range(1, min(13, len(meta)+1))),
            index=5,  # Default to 6 results
            help="Number of results to display"
        )
    
    # Add query guidelines
    with st.expander("📝 Query Guidelines", expanded=False):
        st.markdown("""
        **✅ Good queries:**
        - "person with blonde hair and blue eyes"
        - "smiling woman wearing glasses"
        - "man with beard and mustache"
        - "person in formal suit"
        - "woman with red hair smiling"
        - "person wearing a hat"
        
        **❌ Please avoid:**
        - Special characters: @#$%^&*+=
        - URLs or links
        - Script tags or code
        - Very long descriptions (>500 chars)
        - Obvious inappropriate content
        """)
    
    
   
    
    # Perform search when user enters a query
    if q:
        # Input validation and safety checks
        def validate_query(query):
            """Validate and sanitize user input for safety"""
            import re
            
            # Remove leading/trailing whitespace
            query = query.strip()
            
            # Check if query is empty after stripping
            if not query:
                return None, "Please enter a search query."
            
            # Check minimum length
            if len(query) < 2:
                return None, "Query must be at least 2 characters long."
            
            # Check maximum length (prevent extremely long queries)
            if len(query) > 500:
                return None, "Query is too long. Please keep it under 500 characters."
            
            # Basic content filtering - only block obvious inappropriate content
            # This is minimal to avoid false positives and over-censorship
            inappropriate_patterns = [
                # Only block obvious explicit content patterns
                r'\b(porn|pornographic|xxx)\b',
                # Block obvious hate speech
                r'\b(nazi|terrorist)\b'
            ]
            
            # Check for inappropriate content (minimal filtering)
            for pattern in inappropriate_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return None, "Query contains inappropriate content. Please use descriptive text about appearance and clothing."
            
            # Remove potentially harmful characters but keep normal text
            # Allow letters, numbers, spaces, basic punctuation, and common characters
            safe_query = re.sub(r'[^\w\s\-.,!?()\'":;]', '', query)
            
            # Check if anything meaningful remains after sanitization
            if len(safe_query.strip()) < 2:
                return None, "Query contains invalid characters. Please use only letters, numbers, and basic punctuation."
            
            # Check for suspicious patterns (basic security)
            suspicious_patterns = [
                r'<script', r'javascript:', r'on\w+\s*=', r'data:', r'vbscript:',
                r'file:', r'ftp:', r'http:', r'https:', r'//', r'\\\\'
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return None, "Query contains potentially unsafe content. Please use only descriptive text."
            
            # Additional check for repeated characters (spam-like)
            if re.search(r'(.)\1{4,}', query):  # 5 or more repeated characters
                return None, "Query contains too many repeated characters. Please use normal text."
            
            return safe_query.strip(), None
        
        # Validate the query
        validated_query, error_msg = validate_query(q)
        
        if error_msg:
            st.error(f"❌ {error_msg}")
            st.stop()
        
        # Update the query variable with the validated version
        q = validated_query
        
        # Show the processed query for transparency
        if q != q.strip():  # Only show if query was modified
            st.info(f"🔍 Processed query: '{q}'")
        
        with st.spinner("Searching..."):
            try:
                # Encode the text query using CLIP
                qv = model.encode(q, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
                
                # Calculate cosine similarity between query and all image vectors
                sims = cosine_similarity(qv, vecs).ravel()
                
                # Get top-k most similar images
                top = sims.argsort()[::-1][:k]   
                
                # Display results with enhanced styling
                st.markdown(f"### 🎯 Found {k} Results for: *'{q}'*")
                
                # Display results in rows of 3
                for row_start in range(0, len(top), 3):
                    # Create columns for this row
                    row_cols = st.columns(3)
                    
                    for col_idx in range(3):
                        if row_start + col_idx < len(top):
                            j = top[row_start + col_idx]
                            img_path = meta.loc[j, "path"]
                            
                            with row_cols[col_idx]:
                                if Path(img_path).exists():
                                    # Display images with consistent sizing
                                    st.image(img_path, width=150)
                                    
                                    # Similarity score with progress bar
                                    similarity = float(sims[j])  # Convert to Python float
                                    # Ensure similarity is between 0 and 1 for progress bar
                                    similarity_clamped = max(0.0, min(1.0, similarity))
                                    st.progress(similarity_clamped, text=f"Match: {similarity:.1%}")
                                    
                                    # File info
                                    st.caption(f"📁 {Path(img_path).name}")
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

