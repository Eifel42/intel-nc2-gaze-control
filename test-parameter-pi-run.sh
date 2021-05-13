python3 src/main.py
    --input_file=bin/demo.mp4 \
    --input_type=video \
    --device=MYRIAD \
    --fd=/home/pi/project-controller/model/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml \
    --lr=/home/pi/project-controller/model/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
    --hp=/home/pi/project-controller/model/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml \
    --ge=/home/pi/project-controller/model/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml