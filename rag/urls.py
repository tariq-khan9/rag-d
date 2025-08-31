# rag/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.rag_view, name='rag_view'),
    path('index/', views.rag_view, name='index'),
]