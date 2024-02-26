from flask import Flask, request,render_template
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import Utilities as ult
import glob

app = Flask('Single Image Collision Estimation')

@app.route('/SeverityIdentifier', methods=['POST','GET'])
def run_code():
    if request.method == 'POST':
        Image_path = request.files['filename']
        inputImagePath = 'static/InputImage.jpg'
        Image_path.save(inputImagePath)
        Iou_threshold = float(request.form['IoU_Value'])
        SSIM_threshold = float(request.form['SSIM_Value'])
        img = Image.open(inputImagePath)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        bounding_boxes = ult.Vehicle_Detection(inputImagePath, model)
        whole_images_path = ult.Separating_over_IOU(bounding_boxes, model, img, Iou_threshold)
        test_images = glob.glob(whole_images_path)
        if len(list(test_images)) ==  0:
            resultant_category = dict()
            resultant_category[inputImagePath] = [0,"No Collisions"]
            return render_template("outputPage.html",result = resultant_category)
        pred_categories = ult.Pred_Collision_Category(whole_images_path, SSIM_threshold)
        resultant_category = ult.Collision_Category_Identification(pred_categories)
        return render_template("outputPage.html",result = resultant_category)
    return render_template("inputPage.html")


@app.route('/')
def base_fn():
    return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True)
