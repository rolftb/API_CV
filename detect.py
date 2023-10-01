import sys
from ultralytics import YOLO

def predict(model_path:str, source:str, imgsz:int, conf:float, classes:list):
    model = YOLO(model_path)
    model.predict(source, save=True, imgsz=imgsz, conf=conf,classes=classes)


def main():
    model_path = str(sys.argv[1])
    source = str(sys.argv[2])
    imgsz = int(sys.argv[3])
    conf = float(sys.argv[4])
    classes = list(sys.argv[5])
    print(model_path, source, imgsz, conf, classes)
    predict(model_path, source, imgsz, conf, classes)
if __name__ == "__main__":
    main()
