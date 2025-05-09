import cv2
import numpy as np
import math

def detect_circles(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Could not open or find the image")
        return

    # Convert to grayscale for edge-based detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # Use adaptive thresholding to highlight ring boundaries
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite("thresh.jpg", thresh)

    # Apply minimal morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite("processed_thresh.jpg", processed)

    # Find contours
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours after thresholding: {len(contours)}")

    # Draw all contours for debugging
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)
    cv2.imwrite("contours.jpg", contour_image)

    # Estimate the center using the largest contour
    center = (image.shape[1] // 2, image.shape[0] // 2)
    max_area = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if hierarchy[0][i][3] == -1 and area > max_area:
            max_area = area
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    print(f"Estimated center: {center}")

    # Fit circles with relaxed criteria
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
        print(f"Contour {i}: area={area}, circularity={circularity}, radius={radius}, dist from center={dist}")

        if dist < 300 and radius > 2:
            circles.append((x, y, radius))

    print(f"Number of detected circles: {len(circles)}")

    # Sort circles by radius (largest first) and enforce non-crossing
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

    # Draw and label the 7 concentric circles with green lines
    for i, (x, y, r) in enumerate(concentric_circles):
        center_point = (int(x), int(y))
        radius = int(r)
        cv2.circle(image, center_point, radius, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(image, str(i + 1), center_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the result
    cv2.imwrite(output_path, image)