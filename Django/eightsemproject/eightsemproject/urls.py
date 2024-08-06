"""
URL configuration for eightsemproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from welcome import views 
from welcome.views import welcome
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', welcome, name='welcome'),
    path('register/', views.register, name='register'),
    path('storestudentsinfo/', views.storestudentsinfo, name='storestudentsinfo'),
    path('recordview/', views.recordview, name='recordview'),
    path('detect_faces/', views.detect_faces, name='detect_faces'),
]+ static('/saved_image/', document_root=settings.BASE_DIR / 'saved_image')
