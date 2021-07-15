CUDA_VISIBLE_DEVICES=0,1 python3 IterativePruning.py --weights weights/yolov5_0430.pt --data recycle.yaml --img 1280 --batch-size 8 --crop_aug
# data 파일 수정