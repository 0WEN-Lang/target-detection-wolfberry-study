from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("runs/detect/train-2/weights/best.pt")
    results = model.predict(
        source="datasets/train/images/39878_jpg.rf.00938810463db90f8f08bc08b6af5a76.jpg",
        save=True,
        conf=0.05,
        workers=0
    )
