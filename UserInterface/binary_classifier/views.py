from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# views.py

def binary_classifier(request):
    if request.method == 'POST':
        img = request.FILES['image']
        fs = FileSystemStorage()
        if 'image1_btn' in request.POST:
            filename = fs.save('image1.jpg', img)
            return render(request, 'binary_classifier.html', {'upload1_successful': True})
        if 'image2_btn' in request.POST:
            filename = fs.save('image2.jpg', img)
            return render(request, 'binary_classifier.html', {'upload2_successful': True})
    return render(request, 'binary_classifier.html')

