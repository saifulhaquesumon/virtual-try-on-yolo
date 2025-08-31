import sys
import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv
import requests
import base64
import cv2
import supervision as sv
import numpy as np
import io
from ultralytics import SAM
from inference_sdk import InferenceHTTPClient

# --- Page Configuration ---
st.set_page_config(
    page_title="Virtual Clothing Try-On",
    page_icon="👕",
    layout="wide"
)

# --- Load Environment Variables ---
load_dotenv()

# --- Caching for Model Loading ---
# Cache the model loading to prevent reloading on every interaction
@st.cache_resource
def load_sam_model():
    return SAM("sam_b.pt")

@st.cache_resource
def get_inference_client():
    return InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=os.environ["YOLO_API_KEY"]
    )

# --- Helper Functions ---

def extract_bboxes_and_classes(detection_results):
    """
    Extracts bounding boxes and class names from Roboflow's detection results.
    """
    predictions = detection_results.get("predictions", [])
    bboxes = []
    class_names = []

    for pred in predictions:
        bbox = [
            int(pred["x"] - pred["width"] / 2),   # x_min
            int(pred["y"] - pred["height"] / 2),  # y_min
            int(pred["x"] + pred["width"] / 2),   # x_max
            int(pred["y"] + pred["height"] / 2)   # y_max
        ]
        bboxes.append(bbox)
        class_names.append(pred["class"])
    return bboxes, class_names

def save_binary_masks(segment_result, save_path):
    """
    Saves the first binary mask from segmentation results.
    """
    if not segment_result or not segment_result[0].masks:
        st.error("Segmentation failed or no masks were produced.")
        return False

    # Ensure the save directory exists
    #os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mask = segment_result[0].masks.data[0].cpu().numpy()  # Get the first mask
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    cv2.imwrite(save_path, binary_mask)
    return True

def image_to_base64(image_path):
    """
    Converts an image file from the filesystem to a base64 string.
    """
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# --- Main Application Logic ---

st.title("👕 Virtual Clothing Try-On")
st.markdown("Upload a photo, choose an item of clothing, and describe what you want to change it to!")

# --- Model and Client Initialization ---
try:
    CLIENT = get_inference_client()
    model = load_sam_model()
    SEGMIND_API_KEY = os.environ["SEGMIND_API_KEY"]
except (KeyError, Exception) as e:
    st.error(f"Error initializing services: Make sure your API keys are set in the .env file. Details: {e}")
    st.stop()

# --- File Uploader and User Inputs ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    with col2:
        clothing_choice = st.radio(
            "Which part of the clothing do you want to change?",
            ("Upper Body", "Lower Body"),
            key="clothing_choice"
        )
        prompt = st.text_input("Describe the new clothing:", "A red floral print shirt")
        generate_button = st.button("✨ Change Clothing")

    # --- Processing on Button Click ---
    if generate_button:
        # Define class names for upper and lower body clothing
        UPPER_BODY_CLASSES = ['shirt', 't-shirt', 'top', 'blouse', 'sweater', 'jacket']
        LOWER_BODY_CLASSES = ['pants', 'jeans', 'shorts', 'skirt', 'trousers']

        target_classes = UPPER_BODY_CLASSES if clothing_choice == "Upper Body" else LOWER_BODY_CLASSES

        st.markdown("---")
        st.subheader("Processing Steps")
        progress_cols = st.columns(4)

        try:
            # 1. Object Detection
            with progress_cols[0]:
                with st.spinner('Detecting...'):
                    result = CLIENT.infer(temp_image_path, model_id="main-fashion-wmyfk/1")
                    bboxes, class_names = extract_bboxes_and_classes(result)

                    # Find the target bounding box
                    selected_bbox = None
                    for i, class_name in enumerate(class_names):
                        if class_name in target_classes:
                            selected_bbox = bboxes[i]
                            break

                    if not selected_bbox:
                        st.error(f"Could not detect any '{clothing_choice}' clothing. Please try another image or selection.")
                        st.stop()
                    
                    # Annotate and display the detected image
                    image_cv = cv2.imread(temp_image_path)
                    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    cv2.rectangle(image_rgb, (selected_bbox[0], selected_bbox[1]), (selected_bbox[2], selected_bbox[3]), (0, 255, 0), 2)
                    st.image(image_rgb, caption="1. Detected Object")

            # 2. Segmentation
            with progress_cols[1]:
                with st.spinner('Segmenting...'):
                    segment_result = model(temp_image_path, bboxes=selected_bbox)
                    segmented_img_np = segment_result[0].plot() # Returns numpy array
                    st.image(segmented_img_np, caption="2. Segmented Area")
                    print(temp_image_path)
            # 3. Create Binary Mask
            with progress_cols[2]:
                with st.spinner('Masking...'):
                    mask_path = "mask.png"
                    if save_binary_masks(segment_result, save_path=mask_path):
                        st.image(mask_path, caption="3. Binary Mask")
                    else:
                        st.stop()

            # 4. Inpainting
            with progress_cols[3]:
                with st.spinner('Inpainting...'):
                    url = "https://api.segmind.com/v1/sdxl-inpaint"
                    data = {
                        "image": image_to_base64(temp_image_path),
                        "mask": image_to_base64(mask_path),
                        "prompt": prompt,
                        "negative_prompt": "bad quality, deformed, blurry, pixelated",
                        "scheduler": "DDIM",
                        "num_inference_steps": 30,
                        "guidance_scale": 7.5,
                        "strength": 1.0, # Use 1.0 for full replacement in the masked area
                        "base64": False
                    }
                    headers = {'x-api-key': SEGMIND_API_KEY}
                    response = requests.post(url, json=data, headers=headers)

                    if response.status_code == 200:
                        inpainted_image = Image.open(io.BytesIO(response.content))
                    else:
                        st.error(f"Inpainting failed. API Error: {response.text}")
                        st.stop()

            # --- Final Result ---
            st.markdown("---")
            st.subheader("🎉 Final Result")
            st.image(inpainted_image, caption=f"Result for: '{prompt}'", use_column_width=True)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            
            exc_type, exc_value, exc_traceback = sys.exc_info()
              # Correct: traceback object provided
            st.error(f"Trace: {e.with_traceback(exc_traceback)}")

        finally:
            # Clean up temporary files
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if os.path.exists("mask.png"):
                os.remove("mask.png")