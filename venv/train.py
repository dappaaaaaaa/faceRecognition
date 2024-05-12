from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    model.train(data='C:/Users/ACER/OneDrive/Documents/Kuliah/PKM/FaceRecognitionWithCuda/Dataset/splitData/dataOffline.yaml', epochs=3)


if __name__ == '__main__':
    main()