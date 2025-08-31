# myproject/urls.py
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('rag/', include('rag.urls')),
    path('admin/', admin.site.urls),
]
