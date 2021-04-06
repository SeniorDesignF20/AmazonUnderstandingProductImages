from django.urls import path
from binary_classifier import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.binary_classifier, name='binary_classifier'),
    path('about/', views.about, name='about'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
