import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

min_width_rect = 80
min_height_rect = 80

count_line = 550
offset = 6
counter = 0

# Initialize Substractor
algo = cv2.createBackgroundSubtractorMOG2()

def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 5)

    # Applying on each frame
    img_sub = algo.apply(blur)
    dilated = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_2 = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    dilated_2 = cv2.morphologyEx(dilated_2, cv2.MORPH_CLOSE, kernel)
    counterShape, h = cv2.findContours(dilated_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    cv2.line(frame, (25, count_line), (1200, count_line), (0, 0, 255), 3)

    for i, channel in enumerate(counterShape):
        x, y, w, h = cv2.boundingRect(channel)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)

        if not validate_counter:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Vehicle: "+str(counter), (x, y-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (255, 0, 0), -1)

        for (x, y) in detect:
            if (y < (count_line + offset)) and (y > (count_line - offset)):
                counter += 1
                cv2.line(frame, (25, count_line), (1200, count_line), (0, 255, 0), 3)
                detect.remove((x, y))
                print(f"Vehicle Counter: {str(counter)}")

    cv2.putText(frame, "VEHICLE COUNTER: "+str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    # cv2.imshow('Detector', dilated_2)

    cv2.imshow('Video Original Frame', frame)

    key = cv2.waitKey(100)
    if key == 27 or key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
