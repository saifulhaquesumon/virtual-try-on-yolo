# virtual-try-on-yolo
Detect cloths and change according to user prompt

1. Create a .env file: In the same directory, create a file named .env and add your API keys like this:
==================================================================
YOLO_API_KEY="YOUR_ROBOFLOW_API_KEY"
SEGMIND_API_KEY="YOUR_SEGMIND_API_KEY"

2. Install dependencies: Open your terminal or command prompt and install the required libraries.
===================================================================

pip install streamlit ultralytics supervision numpy opencv-python python-dotenv requests Pillow matplotlib

3. Run the app: In your terminal, navigate to the directory where you saved the files and run the following command:
=========================================================================
streamlit run app.py
Your web browser should open with the running application. Browse a file and click on Generate Button
