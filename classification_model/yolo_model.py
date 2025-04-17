from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")  

results = model.train(data="data/parts", epochs=20, imgsz=64)

model.save("yolo11n-cls-trained.pt")