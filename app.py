from inference_sdk import InferenceHTTPClient
import os
from dotenv import load_dotenv
import requests
import base64
import cv2
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io



load_dotenv()

image_path = "men_shirt.jpg"


CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["YOLO_API_KEY"]
)

result = CLIENT.infer(image_path, model_id="main-fashion-wmyfk/1")

print(result)


#-------------------------------Bounding Box Visualization-------------------------------


# def draw_bounding_boxes(image_path, detection_results):
#     """
#     Draws polished bounding boxes with semi-transparent label backgrounds.
#     """

#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found: {image_path}")

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     predictions = detection_results["predictions"]

#     # xyxy format
#     xyxy = np.array([
#         [
#             pred["x"] - pred["width"] / 2,   # x_min
#             pred["y"] - pred["height"] / 2,  # y_min
#             pred["x"] + pred["width"] / 2,   # x_max
#             pred["y"] + pred["height"] / 2   # y_max
#         ]
#         for pred in predictions
#     ])

#     detections = sv.Detections(
#         xyxy=xyxy,
#         confidence=np.array([pred["confidence"] for pred in predictions]),
#         class_id=np.array([pred["class_id"] for pred in predictions])
#     )

#     labels = [
#         f"{pred['class']} {pred['confidence']*100:.1f}%"
#         for pred in predictions
#     ]

#     # Colors (BGR int tuples)
#     cmap = plt.get_cmap("tab20")
#     colors = [
#         tuple(int(c*255) for c in cmap(i % 20)[:3][::-1])
#         for i in range(len(predictions))
#     ]

#     # Draw boxes
#     box_annotator = sv.BoxAnnotator(thickness=3)
#     annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)

#     # Draw semi-transparent text boxes
#     for i, (x_min, y_min, _, _) in enumerate(xyxy):
#         label = labels[i]
#         color = colors[i]

#         # Text size
#         (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

#         # Rectangle coords
#         x1, y1 = int(x_min), int(y_min) - th - baseline - 4
#         x2, y2 = int(x_min) + tw + 6, int(y_min)

#         # Ensure bounds
#         x1, y1 = max(x1, 0), max(y1, 0)

#         # Overlay for transparency
#         overlay = annotated_image.copy()
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

#         # Add transparency (0.4 = 40%)
#         cv2.addWeighted(overlay, 0.4, annotated_image, 0.6, 0, annotated_image)

#         # Put text
#         cv2.putText(
#             annotated_image,
#             label,
#             (x1 + 3, y2 - 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (255, 255, 255),
#             2
#         )

#     plt.figure(figsize=(10, 10))
#     plt.imshow(annotated_image)
#     plt.axis("off")
#     plt.show()




# # Usage
# draw_bounding_boxes(image_path, result)


#----------------------------

from ultralytics import SAM

# Load a model
model = SAM("sam2.1_b.pt")

# Display model information (optional)
print(model.info())


def extract_bboxes(detection_results):
    """
    Extracts bounding boxes from detection results in [x_min, y_min, x_max, y_max] format.
    """
    predictions = detection_results["predictions"]

    bboxes = [
        [
            int(pred["x"] - pred["width"] / 2),   # x_min
            int(pred["y"] - pred["height"] / 2),  # y_min
            int(pred["x"] + pred["width"] / 2),   # x_max
            int(pred["y"] + pred["height"] / 2)   # y_max
        ]
        for pred in predictions
    ]
    return bboxes

# Get bounding boxes from result
bboxes = extract_bboxes(result)

# Run inference with bboxes prompt
segment_result = model(image_path, bboxes=bboxes[1])

# Assuming `segment_result` is your output
segmented_img = segment_result[0].plot()   # returns numpy array with overlays

plt.imshow(segmented_img)
plt.axis("off")
plt.show()



def save_binary_masks(segment_result, save_dir="masks"):
    """
    Saves binary masks (white object, black background) for each detected object.
    """
    masks = segment_result[0].masks.data.cpu().numpy()  # (N, H, W)

    for i, mask in enumerate(masks):
        binary_mask = (mask > 0.5).astype(np.uint8) * 255  # convert to 0/255

        # Save as PNG
        cv2.imwrite(f"{save_dir}/mask_{i}.png", binary_mask)

        # Optional: show one
        if i == 0:
            plt.imshow(binary_mask, cmap="gray")
            plt.axis("off")
            plt.show()

# Usage
save_binary_masks(segment_result, save_dir="content/masks")
#save_binary_masks(segment_result)


#--------------Inpainting  ----------------------------
# 
# 
#  


# Use this function to convert an image file from the filesystem to base64
def image_file_to_base64(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

# Use this function to fetch an image from a URL and convert it to base64
def image_url_to_base64(image_url):
    response = requests.get(image_url)
    image_data = response.content
    return base64.b64encode(image_data).decode('utf-8')

# Use this function to convert a list of image URLs to base64
def image_urls_to_base64(image_urls):
    return [image_url_to_base64(url) for url in image_urls]

api_key = os.environ["SEGMIND_API_KEY"]
url = "https://api.segmind.com/v1/sdxl-inpaint"

# Request payload
data = {
  "image": image_file_to_base64(image_path),  # Or use image_file_to_base64("IMAGE_PATH")
  "mask": image_file_to_base64("content/masks/mask_0.png"),  # Or use image_file_to_base64("IMAGE_PATH")
  "prompt": "A green Jeans",
  "negative_prompt": "bad quality, painting, blur",
  "samples": 1,
  "scheduler": "DDIM",
  "num_inference_steps": 25,
  "guidance_scale": 7.5,
  "seed": 12467,
  "strength": 0.9,
  "base64": False
}

headers = {'x-api-key': api_key}

response = requests.post(url, json=data, headers=headers)
print(response.content)  # The response is the generated image

#---------------show content--------

# Assuming 'response' variable from the previous cell exists and contains the image bytes
image_bytes = response.content

# Use io.BytesIO to treat the byte string as a file
image_stream = io.BytesIO(image_bytes)

# Open the image using PIL
image = Image.open(image_stream)

# Display the image using matplotlib
plt.imshow(image)
plt.axis('off') # Hide axes
plt.show()