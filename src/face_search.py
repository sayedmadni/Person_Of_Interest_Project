import streamlit as st
from PIL import Image
from pathlib import Path
import random

# ------------------------
# Backend function (Python)
# ------------------------
def get_face_for_query(query: str) -> str:
    """
    This function simulates a backend search.
    It receives a query (e.g., 'smiling') and returns the path
    to a matching image stored locally.
    """
    # Folder with your images
    image_folder = Path("celebratingImages")

    # Simple mapping: define query -> list of matching filenames
    query_map = {
        "smiling": ["smile1.jpg", "angelia.jpeg"],
        "young": ["young1.jpg", "joe-jonas.jpeg"],
        "long hair": ["hat1.jpg", "anne.jpeg"],
        "wearing hat": ["hat1.jpg", "angelia.jpeg"],

        # Add more mappings as needed
    }

    # Get matching images
    matching_files = query_map.get(query.lower(), [])

    if not matching_files:
        return None

    # Pick a random image from matches
    selected_image = random.choice(matching_files)
    image_path = image_folder / selected_image
    if image_path.exists():
        return str(image_path)
    else:
        return None

# ------------------------
# Streamlit frontend
# ------------------------
def main():
    st.title("Face Search 🔍")
    st.write("Search for faces by attributes (e.g., smiling, young, wearing_hat)")

    query = st.text_input("Enter your query:")

    if query:
        image_path = get_face_for_query(query.strip())
        if image_path:
            st.image(Image.open(image_path), caption=f"Result for '{query}'", width=200)
        else:
            st.warning(f"No face found for query '{query}'.")

if __name__ == "__main__":
    main()
