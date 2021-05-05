from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
from pathlib import Path
import sys
from binary_classifier.storage import OverwriteStorage
from PIL import Image

# views.py


def binary_classifier(request):
    media_path = './media/'
    classify_val = 'NaN'
    gradcam_vis1 = ''
    gradcam_vis2 = ''
    box_vis1 = ''
    box_vis2 = ''
    
    if request.method == 'POST' and 'image' in request.FILES:
        img = request.FILES['image']
        fs = OverwriteStorage()

        # check which upload button was clicked
        if 'image1_btn' in request.POST:
            if request.session.has_key('img1_path') and os.path.isfile(media_path + request.session['img1_path']):
                os.remove(media_path + request.session['img1_path'])
                del request.session['img1_path']
            filename = fs.save(img.name, img)
            request.session['img1_path'] = img.name

        elif 'image2_btn' in request.POST:
            if request.session.has_key('img2_path') and os.path.isfile(media_path + request.session['img2_path']):
                os.remove(media_path + request.session['img2_path'])
                del request.session['img2_path']
            filename = fs.save(img.name, img)
            request.session['img2_path'] = img.name

    if request.method == 'POST' and 'classify' in request.POST:
        if not request.session.has_key('img1_path') or not request.session.has_key('img2_path'):
            print("error!")
        else:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            classify_path = str(
                Path(curr_path).parents[1]) + '/model'
            sys.path.insert(1, classify_path)
            from binary_classify import classify
            from gradcam import gradcam
            img1_path = media_path + request.session['img1_path']
            img2_path = media_path + request.session['img2_path']
            print(img1_path)
            print(img2_path)
            classify_val = classify(img1_path, img2_path, classify_path, "small")
            gradcam_vis1, gradcam_vis2, box_vis1, box_vis2 = gradcam(
                img1_path, img2_path, classify_path, "small")
            gradcam_vis1.save('./media/vis1.png', 'PNG')
            gradcam_vis2.save('./media/vis2.png', 'PNG')
            box_vis1.save('./media/box1.png', 'PNG')
            box_vis2.save('./media/box2.png', 'PNG')
            gradcam_vis1 = "./media/vis1.png"
            gradcam_vis2 = "./media/vis2.png"
            box_vis1 = "./media/box1.png"
            box_vis2 = "./media/box2.png"
   
    if request.session.has_key('img1_path') and os.path.isfile('./media/' + request.session['img1_path']):
        img1_path = './media/' + request.session['img1_path']
    else: img1_path = ''
    if request.session.has_key('img2_path') and os.path.isfile('./media/' + request.session['img2_path']):
        img2_path = './media/' + request.session['img2_path']
    else: img2_path = ''
  
    return render(request, 'binary_classifier.html', {'img1_path': img1_path, 'img2_path': img2_path, 
                                                    'classify_val': classify_val, 'vis1': gradcam_vis1, 'vis2': gradcam_vis2, 
                                                    'box1': box_vis1, 'box2': box_vis2})


def about(request):
    return render(request, 'about.html')
