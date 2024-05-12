import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from time import time


#####################################
classID = 1 #0 is Fake dan 1 is Real
outputFolderPath = "C:/Users/ACER/OneDrive/Documents/Kuliah/PKM/FaceRecognitionWithCuda/Dataset/DataCollect"
confidence = 0.8
save = True
blurThreshold = 35 #Larger is more focus

debug = False
offsetPercentageW = 10
offsetPercentageH = 20

camWidth, camHeight = 640, 480
floatingPoint = 6
#####################################

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
detector = FaceDetector()

# Run the loop to continually get frames from the webcam
while True:
    # Read the current frame from the webcam
    # success: Boolean, whether the frame was successfully grabbed
    # img: the captured frame
    success, img = cap.read()

    img = cv2.flip(img, 1)

    # Detect faces in the image
    # img: Updated image
    # bboxs: List of bounding boxes around detected faces
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = [] # Mengingikasikan Nilai True False jika wajah Blur atau tidak
    listInfo = [] # The normalize values and the class name for the label txt file

    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'

            # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = bbox['score'][0]
            # print("besar x : ", x, "besar y : ", y, "besar w ", w,"besar y : ", h)


            # ---------- Cek score ----------
            if score > confidence :

                # ---------- Kotak Lebar Keatas ----------
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)

                # ---------- Kotak Lebar Kebawah ----------
                offsetH = (offsetPercentageH / 100) * w
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                # ---------- Menemukan yang blur ----------
                imgFace = img[y:y+h, x:x+w]
                if imgFace.shape[0] > 0 and imgFace.shape[1] > 0:
                    cv2.imshow("Face", imgFace)
                    blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # ---------- Normalisasi Value ----------
                ih, iw, _ = img.shape #ih = image height, iw = image width, _ = chanel
                xc, yc = x+w/2, y+h/2

                xcn,ycn = round(xc/iw, floatingPoint), round(yc/ih, floatingPoint) #xcn = normalisasi nilai tengah x. xyn = normalisasi nilai tengah y.
                wn,hn = round(w/iw, floatingPoint), round(h/ih, floatingPoint) #wn = normalisasi nilai lebar. hn = normalisasi nilai tinggi.
                # print("titik X : ", xcn,"titik Y: ",ycn, "titik W : ", wn, "titin H", hn)

                # ---------- Untuk Menghilangkan Value di atas 1 ----------
                if xcn < 0 : xcn = 0
                if ycn < 0 : ycn = 0
                if wn < 0 : wn = 0
                if hn < 0 : hn = 0

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                #  ---------- Gambar ----------
                cv2.rectangle(img,(x, y, w, h),(255, 0, 0),3)
                cvzone.putTextRect(img,f'Score : {int(score*100)}% Blur : {blurValue}',(x,y-20), scale = 2, thickness = 3)


        # ---------- Untuk Simpan ----------
        if save:
            print(listBlur)
            print(all(listBlur))
            if all(listBlur) and listBlur != []:
                #---------- Simpan Gambar ----------
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0]+timeNow[1]
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)

                # ---------- Simpan Teks Label File ----------
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()

    # Display the image in a window named 'Image'
    cv2.imshow("Image", img)
    # Wait for 1 millisecond, and keep the window open
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break