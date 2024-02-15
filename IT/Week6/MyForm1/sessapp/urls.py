from django.urls import re_path 
from . import views

urlpatterns = [
re_path(r'^connection/',views.formView, name = 'formView'),
re_path(r'^login/', views.login, name = 'login'),
re_path(r'^logout/', views.logout, name = 'logout'),
]