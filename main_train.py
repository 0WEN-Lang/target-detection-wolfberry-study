from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolo26n.pt")
    model.train(
        data="datasets/data.yaml",
        epochs=100,
        imgsz=640,
        device=0,
        plots=False,
        batch=4,
        amp=False,
        workers=0
    )
