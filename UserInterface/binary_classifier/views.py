from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
from pathlib import Path
import sys
from binary_classifier.storage import OverwriteStorage

# views.py


def binary_classifier(request):
    img1_path = ''
    img2_path = ''
    classify_val = 'NaN'
    gradcam_vis1 = ""
    gradcam_vis2 = ""
    box_vis1 = ""
    box_vis2 = ""

    if request.method == 'POST' and 'image' in request.FILES:
        img = request.FILES['image']
        fs = OverwriteStorage()
        print(request.FILES)
        # check which upload button was clicked
        if 'image1_btn' in request.POST:
            filename = fs.save('image1.jpg', img)
        elif 'image2_btn' in request.POST:
            filename = fs.save('image2.jpg', img)

    if os.path.isfile('./media/image1.jpg'):
        img1_path = './media/image1.jpg'
    if os.path.isfile('./media/image2.jpg'):
        img2_path = './media/image2.jpg'

    if request.method == 'POST' and 'classify' in request.POST:
        curr_path = os.path.dirname(os.path.abspath(__file__))
        classify_path = str(
            Path(curr_path).parents[1]) + '/model'
        sys.path.insert(1, classify_path)
        from binary_classify import classify
        from gradcam import gradcam
        classify_val = classify(img1_path, img2_path, classify_path, "small")
        gradcam_vis1, gradcam_vis2, box_vis1, box_vis2 = gradcam(
            img1_path, img2_path, classify_path, "small")

    return render(request, 'binary_classifier.html', {'img1_path': img1_path, 'img2_path': img2_path, 'classify_val': classify_val, 'vis1': gradcam_vis1, 'vis2': gradcam_vis2, 'box1': box_vis1, 'box2': box_vis2})


def about(request):
    return render(request, 'about.html')
