import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('D:/Project/Algorithm/RTDETR-main/rt-detr/DUT-Anti-UAV/rtdetr-anti-r18/weights/best.pt')
    model.val(data='D:/Project/Algorithm/RTDETR-main/ultralytics/cfg/datasets/UAV/anti-uav.yaml',
              split='val',  # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=1,
              save_json=True,  # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )

# 20.8ms
# 50.7ms
# 35.0ms