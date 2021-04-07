from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
from binary_classifier.storage import OverwriteStorage

# views.py

def binary_classifier(request):
    img1_path = ''
    img2_path = ''

    if request.method == 'POST' and 'image' in request.FILES:
        img = request.FILES['image']
        fs = OverwriteStorage()

        # check which upload button was clicked
        if 'image1_btn' in request.POST:
            filename = fs.save('image1.jpg', img)
        elif 'image2_btn' in request.POST:
            filename = fs.save('image2.jpg', img)

        return redirect('binary_classifier')

    if os.path.isfile('./media/image1.jpg'):
            img1_path = './media/image1.jpg'
    if os.path.isfile('./media/image2.jpg'):
            img2_path = './media/image2.jpg'

    return render(request, 'binary_classifier.html', {'img1_path': img1_path, 'img2_path': img2_path})

def about(request):
    return render(request, 'about.html')

