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
    .main {
        background-color: black;
    }
    .stButton button {
        background-color: #0066cc;
        color: white;
    }
    .stButton button:hover {
        background-color: #004d99;
        color: white;
    }
    h1, h2, h3 {
        color: #0066cc;
    }
    .stRadio label {
        margin-right: 20px;
        font-weight: bold;
    }
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

# Save and load training data to/from disk so training happens only once.
def save_training_data(embeddings, names, filename="training_data.pkl"):
    data = {"embeddings": embeddings, "names": names}
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_training_data_from_disk(filename="training_data.pkl"):
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
    if 'training_folder' not in st.session_state:
        st.session_state.training_folder = ""
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Training"

# --------------------------
# CACHED FUNCTION TO LOAD TRAINING DATA FROM A FOLDER
@st.cache_data(show_spinner=False)
def get_training_data(folder_path):
    face_analysis_app = load_face_analysis()
    known_face_embeddings = []
    known_face_names = []
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    for image_path in image_files:
        img = cv2.imread(image_path)
        if img is not None:
            faces = face_analysis_app.get(img)
            if faces:
                for face in faces:
                    if face.embedding is not None:
                        normalized_embedding = normalize_embedding(face.embedding)
                        known_face_embeddings.append(normalized_embedding)
                        name = os.path.splitext(os.path.basename(image_path))[0]
                        known_face_names.append(name)
    success = True if known_face_embeddings else False
    return known_face_embeddings, known_face_names, success

# Optionally, allow uploading training images individually.
def load_training_images_from_uploads(uploaded_files):
    face_analysis_app = load_face_analysis()
    known_face_embeddings = []
    known_face_names = []
    
    loading_placeholder = display_loading_screen("Processing uploaded training images...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress(int((i + 1) / len(uploaded_files) * 100))
        status_text.text(f"Processing image {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        img = cv2.imread(tmp_path)
        if img is not None:
            faces = face_analysis_app.get(img)
            if faces:
                for face in faces:
                    if face.embedding is not None:
                        normalized_embedding = normalize_embedding(face.embedding)
                        known_face_embeddings.append(normalized_embedding)
                        name = os.path.splitext(uploaded_file.name)[0]
                        known_face_names.append(name)
        
        os.unlink(tmp_path)
    
    progress_bar.empty()
    status_text.empty()
    loading_placeholder.empty()
    
    return known_face_embeddings, known_face_names

# Recognize faces in attendance images
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
                
                face_data = {
                    "name": name,
                    "embedding": embedding,
                    "image_name": uploaded_file.name
                }
                recognized_data.append(face_data)
        
        os.unlink(tmp_path)
    
    progress_bar.empty()
    status_text.empty()
    loading_placeholder.empty()
    
    return recognized_data

# Save attendance CSV
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
    
    # Try loading training data from disk if not already trained in session.
    if not st.session_state.training_complete:
        embeddings, names = load_training_data_from_disk()
        if embeddings is not None and names is not None:
            st.session_state.known_face_embeddings = embeddings
            st.session_state.known_face_names = names
            st.session_state.training_complete = True
            st.success("Loaded training data from disk!")
    
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
        st.header("Train the Recognition System")
        
        # Only ask for training folder if not already trained
        if not st.session_state.training_complete:
            training_folder = st.text_input(
                "Enter the Training Images Folder Path", 
                value=st.session_state.training_folder,
                help="Provide the full path to the folder containing training images. Each image filename should be the student's name."
            )
            if training_folder != st.session_state.training_folder:
                st.session_state.training_folder = training_folder
                st.session_state.training_complete = False
            
            if training_folder:
                if os.path.exists(training_folder):
                    image_files_count = len(glob.glob(os.path.join(training_folder, "*.jpg"))) + \
                                        len(glob.glob(os.path.join(training_folder, "*.jpeg"))) + \
                                        len(glob.glob(os.path.join(training_folder, "*.png")))
                    st.info(f"Found {image_files_count} images in the folder.")
                    
                    with st.spinner("Loading training images from folder..."):
                        embeddings, names, success = get_training_data(training_folder)
                        if success:
                            st.session_state.known_face_embeddings = embeddings
                            st.session_state.known_face_names = names
                            st.session_state.training_complete = True
                            save_training_data(embeddings, names)  # Save data for future reloads
                            st.success(f"Processed {len(names)} faces from the training folder!")
                            st.session_state.current_tab = "Take Attendance"
                            st.rerun()
                        else:
                            st.error("No valid images found in the folder.")
                else:
                    st.warning(f"Folder does not exist: {training_folder}")
            
            st.divider()
            
            st.subheader("Or Upload Training Images Individually")
            uploaded_training_files = st.file_uploader(
                "Upload training images (JPG, JPEG, PNG)", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True,
                key="training_upload"
            )
            if uploaded_training_files:
                with st.spinner("Processing uploaded training images..."):
                    embeddings, names = load_training_images_from_uploads(uploaded_training_files)
                    st.session_state.known_face_embeddings = embeddings
                    st.session_state.known_face_names = names
                    st.session_state.training_complete = True
                    save_training_data(embeddings, names)
                    st.success(f"Processed {len(names)} faces from the uploaded images!")
                    st.session_state.current_tab = "Take Attendance"
                    st.rerun()
        else:
            st.subheader("Training Data Already Loaded")
            st.write(f"Total trained students: {len(st.session_state.known_face_names)}")
            import pandas as pd
            unique_names = pd.DataFrame({"Student Name": list(set(st.session_state.known_face_names))})
            st.dataframe(unique_names, width=400)
    
    # ----------------- Attendance Tab -----------------
    elif st.session_state.current_tab == "Take Attendance":
        st.header("Take Attendance")
        
        if not st.session_state.training_complete:
            st.warning("Please train the system with student images first (go to Training tab)")
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
                    st.rerun()
            
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
                st.rerun()
            
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
                    st.rerun()
            
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
