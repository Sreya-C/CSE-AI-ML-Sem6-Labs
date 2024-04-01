from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^first/', views.firstpage, name='firstpage'),
    re_path(r'^second/', views.secondpage, name='secondpage'),
]
