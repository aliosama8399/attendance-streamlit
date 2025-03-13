import os
import cv2
import csv
import pickle
import numpy as np
import logging
import tempfile
from datetime import datetime
import streamlit as st
from insightface.app import FaceAnalysis
import glob

# Set page config for a clean layout
st.set_page_config(
    page_title="Attendance System",
    page_icon="📊",
    layout="wide"
)

# Configure logging
logging.getLogger('insightface').setLevel(logging.ERROR)

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: black; }
    .stButton button { background-color: #0066cc; color: white; }
    .stButton button:hover { background-color: #004d99; color: white; }
    h1, h2, h3 { color: #0066cc; }
    .stRadio label { margin-right: 20px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Display a loading animation
def display_loading_screen(text="Processing..."):
    with st.spinner(text):
        loading_placeholder = st.empty()
        loading_placeholder.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 40px 0;">
            <div style="border: 8px solid #f3f3f3; border-top: 8px solid #0066cc; border-radius: 50%; width: 60px; height: 60px; animation: spin 2s linear infinite;"></div>
            <p style="margin-top: 20px; font-size: 18px; color: #0066cc;">{text}</p>
        </div>
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """, unsafe_allow_html=True)
        return loading_placeholder

# Initialize the face analysis model (cached for performance)
@st.cache_resource
def load_face_analysis():
    providers = ['CPUExecutionProvider']
    face_app = FaceAnalysis(providers=providers)
    face_app.prepare(ctx_id=0, det_size=(224, 224))
    return face_app

# Normalize an embedding vector
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

# Save training data (embeddings and names) to disk as a pickle file
def save_training_data(embeddings, names, filename):
    data = {"embeddings": embeddings, "names": names}
    with open(filename, "wb") as f:
        pickle.dump(data, f)

# Load training data from disk (from a pickle file)
def load_training_data_from_disk(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data["embeddings"], data["names"]
    return None, None

# Set up session state variables once per session
def initialize_session_state():
    if 'known_face_embeddings' not in st.session_state:
        st.session_state.known_face_embeddings = []
    if 'known_face_names' not in st.session_state:
        st.session_state.known_face_names = []
    if 'recognized_faces' not in st.session_state:
        st.session_state.recognized_faces = []
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'attendance_processed' not in st.session_state:
        st.session_state.attendance_processed = False
    if 'training_set_name' not in st.session_state:
        st.session_state.training_set_name = ""
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Training"

# --------------------------
# CACHED FUNCTION TO LOAD TRAINING DATA FROM Uploaded Files
@st.cache_data(show_spinner=False)
def get_training_data_from_uploads(uploaded_files):
    face_analysis_app = load_face_analysis()
    known_face_embeddings = []
    known_face_names = []
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is not None:
            faces = face_analysis_app.get(img)
            if faces:
                for face in faces:
                    if face.embedding is not None:
                        normalized_embedding = normalize_embedding(face.embedding)
                        known_face_embeddings.append(normalized_embedding)
                        name = os.path.splitext(uploaded_file.name)[0]
                        known_face_names.append(name)
    success = True if known_face_embeddings else False
    return known_face_embeddings, known_face_names, success

# Optionally, allow uploading training images individually (alternative method)
def load_training_images_from_uploads(uploaded_files):
    # This function is similar to get_training_data_from_uploads but without caching
    return get_training_data_from_uploads(uploaded_files)

# Recognize faces in the provided attendance images
def recognize_faces(uploaded_files, known_face_embeddings, known_face_names):
    face_analysis_app = load_face_analysis()
    recognized_data = []
    loading_placeholder = display_loading_screen("Analyzing images for attendance...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress(int((i + 1) / len(uploaded_files) * 100))
        status_text.text(f"Analyzing image {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        img = cv2.imread(tmp_path)
        if img is not None:
            faces = face_analysis_app.get(img)
            for face in faces:
                if face.embedding is None:
                    continue
                embedding = normalize_embedding(face.embedding)
                name = "Unknown"
                min_dist = float("inf")
                for known_embedding, known_name in zip(known_face_embeddings, known_face_names):
                    dist = np.linalg.norm(embedding - known_embedding)
                    if dist < min_dist:
                        min_dist = dist
                        name = known_name if dist < 1.2 else "Unknown"
                face_data = {"name": name, "embedding": embedding, "image_name": uploaded_file.name}
                recognized_data.append(face_data)
        os.unlink(tmp_path)
    progress_bar.empty()
    status_text.empty()
    loading_placeholder.empty()
    return recognized_data

# Save recognized names to CSV for export
def save_recognized_names_to_csv(recognized_names, course_name, hall):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{course_name}_{hall}_{timestamp}.csv"
    with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', suffix='.csv') as tmp_file:
        csv_writer = csv.writer(tmp_file)
        csv_writer.writerow(["Student Name", "Course", "Hall", "Timestamp"])
        for name in recognized_names:
            csv_writer.writerow([name, course_name, hall, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    with open(tmp_file.name, 'r') as f:
        csv_data = f.read()
    return csv_data, csv_filename

# --------------------------
# MAIN APP
def main():
    initialize_session_state()
    
    # Create directory for training sets if it doesn't exist.
    training_sets_dir = "training_sets"
    if not os.path.exists(training_sets_dir):
        os.makedirs(training_sets_dir)
    
    # Header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRVnYQHsVGCaZGazOdtxmDh1QjypY4iixJIPQ&s", width=120)
    with col2:
        st.title("Facial Recognition Attendance System")
        st.markdown("<p style='color:#0066cc'>Record attendance automatically using facial recognition</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation tabs
    tab_options = ["Training", "Take Attendance", "Results"]
    current_index = tab_options.index(st.session_state.current_tab) if st.session_state.current_tab in tab_options else 0
    selected_tab = st.radio("", tab_options, index=current_index, horizontal=True)
    st.session_state.current_tab = selected_tab
    
    # ----------------- Training Tab -----------------
    if st.session_state.current_tab == "Training":
        st.header("Manage Training Sets")
        st.write("Either create a new training set by uploading training images or load an existing one.")
        mode = st.radio("Select Option", ["New Training Set", "Load Existing Training Set"])
        
        if mode == "New Training Set":
            # Instead of a folder path, let the user upload multiple training images.
            uploaded_training_files = st.file_uploader(
                "Upload training images (JPG, JPEG, PNG)", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True,
                key="new_training_upload"
            )
            training_set_name = st.text_input("Enter Training Set Name (e.g., AI_first_year)", value=st.session_state.training_set_name)
            if st.button("Process New Training Set"):
                if uploaded_training_files and training_set_name:
                    with st.spinner("Processing training images..."):
                        embeddings, names, success = get_training_data_from_uploads(uploaded_training_files)
                        if success:
                            st.session_state.known_face_embeddings = embeddings
                            st.session_state.known_face_names = names
                            st.session_state.training_complete = True
                            st.session_state.training_set_name = training_set_name
                            # Save the training data as a pickle file
                            pkl_filename = os.path.join(training_sets_dir, f"{training_set_name}.pkl")
                            save_training_data(embeddings, names, pkl_filename)
                            st.success(f"Training set '{training_set_name}' processed and saved!")
                            st.session_state.current_tab = "Take Attendance"
                            st.experimental_rerun()
                        else:
                            st.error("No valid images found in the uploads.")
                else:
                    st.error("Please upload training images and provide a training set name.")
        
        elif mode == "Load Existing Training Set":
            available_sets = [os.path.splitext(f)[0] for f in os.listdir(training_sets_dir) if f.endswith('.pkl')]
            if available_sets:
                selected_set = st.selectbox("Select Training Set", available_sets)
                if st.button("Load Selected Training Set"):
                    pkl_filename = os.path.join(training_sets_dir, f"{selected_set}.pkl")
                    embeddings, names = load_training_data_from_disk(pkl_filename)
                    if embeddings is not None and names is not None:
                        st.session_state.known_face_embeddings = embeddings
                        st.session_state.known_face_names = names
                        st.session_state.training_complete = True
                        st.session_state.training_set_name = selected_set
                        st.success(f"Loaded training set '{selected_set}'!")
                        st.session_state.current_tab = "Take Attendance"
                        st.experimental_rerun()
                    else:
                        st.error("Failed to load training set.")
            else:
                st.info("No existing training sets available. Please create a new training set.")
        
        if st.session_state.training_complete:
            st.subheader("Training Results")
            st.write(f"Training Set: {st.session_state.training_set_name}")
            st.write(f"Total trained students: {len(st.session_state.known_face_names)}")
            import pandas as pd
            unique_names = pd.DataFrame({"Student Name": list(set(st.session_state.known_face_names))})
            st.dataframe(unique_names, width=400)
    
    # ----------------- Attendance Tab -----------------
    elif st.session_state.current_tab == "Take Attendance":
        st.header("Take Attendance")
        if not st.session_state.training_complete:
            st.warning("Please load a training set first (go to Training tab)")
        else:
            st.write("Upload class images to mark attendance.")
            uploaded_attendance_files = st.file_uploader(
                "Upload class images (JPG, JPEG, PNG)", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True,
                key="attendance_files"
            )
            col1, col2 = st.columns(2)
            with col1:
                course_name = st.text_input("Course Name")
            with col2:
                hall = st.text_input("Hall/Location")
            if course_name:
                st.session_state.course_name = course_name
            if hall:
                st.session_state.hall = hall
            if uploaded_attendance_files and course_name and hall and not st.session_state.attendance_processed:
                with st.spinner("Processing attendance images..."):
                    recognized_data = recognize_faces(
                        uploaded_attendance_files, 
                        st.session_state.known_face_embeddings, 
                        st.session_state.known_face_names
                    )
                    st.session_state.recognized_faces = recognized_data
                    st.session_state.attendance_processed = True
                    st.success(f"Processed {len(recognized_data)} faces!")
                    st.session_state.current_tab = "Results"
                    st.experimental_rerun()
            if st.session_state.recognized_faces:
                import pandas as pd
                attendance_data = []
                for i, face in enumerate(st.session_state.recognized_faces):
                    attendance_data.append({
                        "ID": i+1,
                        "Name": face["name"],
                        "Image": face["image_name"],
                        "Status": "Recognized" if face["name"] != "Unknown" else "Unknown"
                    })
                df = pd.DataFrame(attendance_data)
                st.dataframe(df)
    
    # ----------------- Results Tab -----------------
    elif st.session_state.current_tab == "Results":
        st.header("Attendance Results")
        if not st.session_state.recognized_faces:
            st.info("No attendance data available. Please take attendance first.")
        else:
            st.subheader("Attendance Summary")
            recognized_names = [face["name"] for face in st.session_state.recognized_faces]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Students", len(set(recognized_names)))
            with col2:
                st.metric("Recognized Faces", len([name for name in recognized_names if name != "Unknown"]))
            with col3:
                unknown_count = len([name for name in recognized_names if name == "Unknown"])
                st.metric("Unrecognized Faces", unknown_count)
            st.markdown("### Edit Attendance Sheet")
            updated_names = {}
            for i, face in enumerate(st.session_state.recognized_faces):
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.write(f"Image: {face['image_name']}")
                with col2:
                    new_name = st.text_input("Name", value=face["name"], key=f"edit_rec_{i}")
                    updated_names[i] = new_name
            if st.button("Save Changes"):
                for i, new_name in updated_names.items():
                    st.session_state.recognized_faces[i]["name"] = new_name
                st.success("Attendance sheet updated successfully!")
                st.experimental_rerun()
            st.markdown("### Add New Entry")
            new_entry_name = st.text_input("New Student Name", key="new_entry")
            if st.button("Add Entry"):
                if new_entry_name:
                    st.session_state.recognized_faces.append({
                        "name": new_entry_name,
                        "embedding": None,
                        "image_name": "Manual Entry"
                    })
                    st.success(f"Added new entry: {new_entry_name}")
                    st.experimental_rerun()
            if st.button("Export to CSV"):
                course = st.session_state.get("course_name", "Course")
                location = st.session_state.get("hall", "Hall")
                updated_recognized_names = [face["name"] for face in st.session_state.recognized_faces]
                csv_data, csv_filename = save_recognized_names_to_csv(updated_recognized_names, course, location)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv"
                )

if __name__ == '__main__':
    main()
