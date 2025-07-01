import os
from flask import Flask, request, render_template
import sys
from ultralytics import YOLO
from PIL import Image
import numpy as np
from datetime import datetime

app = Flask(__name__)

# 关闭浏览器缓存
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# 设置上传与检测目录
UPLOAD_FOLDER = 'static/uploads'
DETECT_FOLDER = 'static/detections'
MODELS_FOLDER = 'static/models'  # 新增模型上传目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_detect():
    if request.method == 'POST':
        # 获取客户端上传的图片
        image_file = request.files["image"]
        # 获取客户端上传的模型
        model_file = request.files["model"]

        if image_file and model_file:
            # 处理上传的图片
            img_filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + image_file.filename
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], img_filename)
            detect_path = os.path.join(app.config["DETECT_FOLDER"], img_filename)
            image_file.save(upload_path)

            # 处理上传的模型
            model_filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + model_file.filename
            model_path = os.path.join(app.config["MODELS_FOLDER"], model_filename)
            model_file.save(model_path)

            try:
                # 使用上传的模型进行目标检测
                model = YOLO(model_path)

                # 进行目标检测
                results = model(upload_path)

                # 绘制检测结果图像并保存
                result_img_array = results[0].plot()
                result_pil = Image.fromarray(result_img_array)
                result_pil.save(detect_path)

                # 提取检测框信息（标签 + 置信度）
                detections = []
                boxes = results[0].boxes
                if boxes is not None and boxes.cls.numel() > 0:
                    for cls_id, conf in zip(boxes.cls, boxes.conf):
                        class_name = model.names[int(cls_id)]
                        confidence = round(float(conf) * 100, 2)
                        detections.append(f"{class_name}: {confidence}%")
                else:
                    detections.append("No objects detected.")

                return render_template(
                    'index.html',
                    prediction="Detection Complete",
                    detections=detections,
                    image_path=f"detections/{img_filename}"
                )
            except Exception as e:
                return render_template(
                    'index.html',
                    prediction=f"Error: {str(e)}",
                    detections=[],
                    image_path=None
                )

    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)