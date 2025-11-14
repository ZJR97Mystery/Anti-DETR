import warnings

warnings.filterwarnings('ignore')

import wandb

wandb.login(key='f6e5c5252552863d8732cab1533e58b5617e983f')

from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('D:/Project/Algorithm/RTDETR-main/ultralytics/cfg/models/yolo-detr/yolov8n-detr.yaml')
    # model.load('D:/Project/Algorithm/RTDETR-main/weights/rtdetr-r18.pt')  # loading pretrain weights
    model.train(data='D:/Project/Algorithm/RTDETR-main/ultralytics/cfg/datasets/UAV/anti-uav.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=0,
                workers=4,  # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                device='0',  # 指定显卡和多卡训练问题
                # patience=30,  # set 0 to close early stop.
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                amp=False,  # close amp
                project='rt-detr',
                # fraction=0.2,
                name='yolo-detr',
                lr0=0.01,
                cos_lr=True
                )

    # model.val(data='D:/Project/Algorithm/RTDETR-main/ultralytics/cfg/datasets/UAV/Anti-UAV.yaml',
    #           imgsz=640,
    #           batch=4,
    #           close_mosaic=0,
    #           workers=4,  # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #           device='0',  # 指定显卡和多卡训练问题
    #           )


# --conf-thres 如果检测框重复可以提高置信区间，如从默认的0.25提高到0.3
# --iou-thres 如果检测框重复可以降低iou，如从默认的0.45降低到0.2