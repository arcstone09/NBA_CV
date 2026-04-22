import cv2

cap = cv2.VideoCapture("./data/raw_data/0022400355_142.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

print(fps)