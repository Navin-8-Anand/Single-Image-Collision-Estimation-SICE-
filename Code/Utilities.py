from skimage.metrics import structural_similarity as ssim
from PIL import Image
from itertools import repeat
import cv2
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import pandas

def Vehicle_Detection(img_path, model):
    img = Image.open(img_path)
    detection = model(img)
    bounding_boxes = detection.xyxy[0].tolist()

    return bounding_boxes

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def Separating_over_IOU(bounding_boxes, model, image, threshold_value_iou):
    intersection_dict = {}
    bounding_box_val = []
    for i in range(len(bounding_boxes)):
        for j in range(i + 1, len(bounding_boxes)):
            boxA = bounding_boxes[i]
            boxB = bounding_boxes[j]
            class_index_A = int(boxA[5])
            class_name_A = model.names[class_index_A]
            class_index_B = int(boxB[5])
            class_name_B = model.names[class_index_B]
            intersection = bb_intersection_over_union(boxA, boxB)
            if intersection > threshold_value_iou and class_name_A == 'car' and class_name_B == 'car':
                intersection = format(intersection, '.3f')
                intersection_dict[intersection] = (boxA,boxB)

    for value in intersection_dict.values():
        bounding_box_val.append(value)
    
    for i, whole_bbox in enumerate(bounding_box_val):
        box1 = whole_bbox[0]
        box2 = whole_bbox[1]
        box1 = [int(co_ord) for co_ord in box1]
        box2 = [int(co_ord) for co_ord in box2]
    
        x_min = min(box1[0], box2[0])
        y_min = min(box1[1], box2[1])
        x_max = max(box1[2], box2[2])
        y_max = max(box1[3], box2[3])

        whole_image = image.crop((x_min, y_min, x_max, y_max))
        whole_image.save(f"static\\Vehicle_Localization_Results\\Image_1_{i}.jpg")
    whole_image_path = "static\\Vehicle_Localization_Results\\*"
    return whole_image_path

def find_mean_ssim_value(image_1_path,image_2_path,category_dict,image_category_dict,threshold):
    img1 = cv2.imread(image_1_path)
    img2 = cv2.imread(image_2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    curr_category = image_2_path.split('\\')[-2]
    curr_image = image_2_path.split('\\')[-1].split('.')[0]
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    
    ssim_value = ssim(img1, img2)
    ssim_value = round(ssim_value, 2)
    if curr_category in category_dict and ssim_value > threshold:
        curr_dict = category_dict[curr_category]
        curr_dict[curr_image] = ssim_value
        category_dict[curr_category] = curr_dict
    elif ssim_value > threshold:
        category_dict[curr_category] = {curr_image:ssim_value}
    return category_dict

def Pred_Collision_Category(test_images_path, threshold_value_ssim):
    test_images = glob.glob(test_images_path)
    final_dict = {}
    for image_1_path in test_images:
        image_2_path_list = glob.glob("static\\Types of Collisions\\*\\*") 
        category_dict = {}
        image_category_dict = {}
        ssim_values = list(map(find_mean_ssim_value,repeat(image_1_path),image_2_path_list,repeat(category_dict),repeat(image_category_dict),repeat(threshold_value_ssim)))
        image_id = image_1_path.split('\\')[-1]
        final_dict[image_id] = ssim_values[-1]
    return final_dict

def Collision_Detection(ssim_values_dict):
    mean_values = dict()
    std_values = dict()
    for key in ssim_values_dict.keys():
        values = list(ssim_values_dict[key].values())
        mean = round(np.mean(values),2)
        mean_values[key] = mean
    max_value = max(mean_values.values())
    same_value = list()
    for key, value in mean_values.items():
        if value == max_value:
            same_value.append(key)
    
    for key in same_value:
        values = list(ssim_values_dict[key].values())
        std_dev = round(np.std(values),2)
        std_values[key] = std_dev
    
    minimum_values = list()
    if len(same_value) > 1:
        for i in same_value:
            minimum_values.append(std_values[i])
        std_min = min(minimum_values)
        category = list()
        for key, value in std_values.items():
            if value == std_min:
                category.append(key)
        return category
    else:
        return same_value

def find_max_correlation_image(category,input_dict,image_id):
    max_value  = max(list(input_dict[image_id][category].values()))
    image_id = list(input_dict[image_id][category].keys()) [list(input_dict[image_id][category].values()).index(max_value)]
    return image_id, max_value

def find_severity(severity_level,ssim_value):
    #Formula Starts
    pred_severity = (severity_level * ssim_value)
    #Formula Ends
    
    return pred_severity

def Collision_Category_Identification(pred_categories):
    pred_dict = {}
    corr_dict = {}
    severity_dict = {}
    impact_factor = {}
    for key in pred_categories:
        predicted_category = Collision_Detection(pred_categories[key])
        pred_dict[key] = predicted_category
    
    for key in pred_dict.keys():
        pred_category = pred_dict[key][0]
        image_id, max_value = find_max_correlation_image(pred_category,pred_categories,key)   
        corr_dict[key] = [pred_category,image_id,max_value]

    for key in corr_dict.keys():
        values = corr_dict[key]
        category = values[0]
        if category != 'No Collision':
            data_frame = pandas.read_excel("static\\Severity_Level.xlsx", sheet_name = category)
            severity_level = data_frame[data_frame['Image_Id'] == values[1]]['Severity_Level'].iloc[0]
            pred_severity = find_severity(severity_level,values[2])
            severity_dict[key] = [pred_severity,category]
        else:
            severity_dict[key] = [0,category]
    return severity_dict
