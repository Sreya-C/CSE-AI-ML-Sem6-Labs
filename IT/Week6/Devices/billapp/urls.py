from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^selectpage/', views.selectpage, name='selectpage'),
    re_path(r'^billpage/', views.billpage, name='billpage'),
]