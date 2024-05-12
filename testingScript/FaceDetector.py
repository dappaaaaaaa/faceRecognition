from cvzone.FaceDetectionModule import FaceDetector
import cv2

offsetPercentageW = 10
offsetPercentageH = 20

cap = cv2.VideoCapture(0)

# Initialize the FaceDetector object
# minDetectionCon: Minimum detection confidence threshold
# modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Run the loop to continually get frames from the webcam
while True:
    # Read the current frame from the webcam
    # success: Boolean, whether the frame was successfully grabbed
    # img: the captured frame
    success, img = cap.read()

    # Flip the frame horizontally
    img = cv2.flip(img, 1)

    # Detect faces in the image
    # img: Updated image
    # bboxs: List of bounding boxes around detected faces
    img, bboxs = detector.findFaces(img, draw=False)

    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'

            # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']

            offsetW = (offsetPercentageW / 100) * w

            x = int(x - offsetW)
            w = int(w + offsetW * 2)

            cv2.rectangle(img,(x, y, w, h),(255, 0, 0),3)
    # Display the image in a window named 'Image'
    cv2.imshow("Image", img)
    # Wait for 1 millisecond, and keep the window open
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break