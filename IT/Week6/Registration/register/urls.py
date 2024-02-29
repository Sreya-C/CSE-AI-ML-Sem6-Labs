from django.urls import re_path
from . import views  
  
urlpatterns = [  
    re_path(r'^registerpage/', views.register, name="register"),
    re_path(r'^success/', views.success, name="success"),
#    re_path(r'^home/', views.home, name="home"),
]