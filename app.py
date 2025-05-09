import streamlit as st
import cv2
import numpy as np
import math
from io import BytesIO
from PIL import Image

# CSS for professional styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        text-align: center;
        color: #2c3e50;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .image-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 30px;
    }
    .image-caption {
        text-align: center;
        color: #7f8c8d;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def detect_circles(image):
    if image is None:
        return None, None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(processed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)

    center = (image.shape[1] // 2, image.shape[0] // 2)
    max_area = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if hierarchy[0][i][3] == -1 and area > max_area:
            max_area = area
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

    circles = []
    for i, contour in enumerate(contours):
        if len(contour) < 5:
            continue
        area = cv2.contourArea(contour)
        if area < 50:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.3:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        dist = math.sqrt((x - center[0])**2 + (y - center[1])**2)

        if dist < 300 and radius > 2:
            circles.append((x, y, radius))

    concentric_circles = []
    if circles:
        circles.sort(key=lambda x: x[2], reverse=True)
        concentric_circles.append(circles[0])
        min_spacing = 2
        for i in range(1, len(circles)):
            curr = circles[i]
            is_concentric = True
            for prev in concentric_circles:
                dist = math.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                if dist > 300 or curr[2] >= prev[2] - min_spacing:
                    is_concentric = False
                    break
            if is_concentric and len(concentric_circles) < 7:
                concentric_circles.append(curr)

        while len(concentric_circles) < 7 and circles:
            last = concentric_circles[-1]
            new_radius = last[2] - min_spacing * 2
            if new_radius > 2:
                concentric_circles.append((last[0], last[1], new_radius))
            else:
                break
        if len(concentric_circles) > 7:
            concentric_circles = concentric_circles[:7]

    output_image = image.copy()
    for i, (x, y, r) in enumerate(concentric_circles):
        center_point = (int(x), int(y))
        radius = int(r)
        cv2.circle(output_image, center_point, radius, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(output_image, str(i + 1), center_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return output_image, thresh, processed, contour_image

def image_to_bytes(image):
    _, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()

# Streamlit app
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">Circle Detection Application</div>', unsafe_allow_html=True)

# Display before and after images for teaching
col1, col2 = st.columns(2)
with col1:
    before_image = cv2.imread(r"data\before_detection.jpg")
    if before_image is not None:
        before_rgb = cv2.cvtColor(before_image, cv2.COLOR_BGR2RGB)
        st.image(before_rgb, use_column_width=True)
        st.markdown('<div class="image-caption">Before Detection</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="image-caption">Before Detection Image Not Found</div>', unsafe_allow_html=True)
with col2:
    after_image = cv2.imread(r"data\after_detection.png")
    if after_image is not None:
        after_rgb = cv2.cvtColor(after_image, cv2.COLOR_BGR2RGB)
        st.image(after_rgb, use_column_width=True)
        st.markdown('<div class="image-caption">After Detection</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="image-caption">After Detection Image Not Found</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load uploaded image into memory
    image_bytes = uploaded_file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, use_column_width=True)

    # Button to trigger processing
    if st.button("Detect Circles"):
        output_image, thresh, processed, contour_image = detect_circles(image)

        if output_image is not None:
            # Display output image
            output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            st.image(output_rgb, use_column_width=True)
            # Download button for output
            st.download_button(
                label="Download Output",
                data=image_to_bytes(output_image),
                file_name="output.png",
                mime="image/png"
            )

        # Display debug images
        debug_images = [
            (thresh, "Threshold", "thresh.png"),
            (processed, "Processed Threshold", "processed_thresh.png"),
            (contour_image, "Contours", "contours.png")
        ]
        for debug_img, label, filename in debug_images:
            if debug_img is not None:
                debug_rgb = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB) if len(debug_img.shape) == 2 else cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                st.image(debug_rgb, use_column_width=True)
                st.download_button(
                    label=f"Download {label}",
                    data=image_to_bytes(debug_img),
                    file_name=filename,
                    mime="image/png"
                )

st.markdown('</div>', unsafe_allow_html=True)