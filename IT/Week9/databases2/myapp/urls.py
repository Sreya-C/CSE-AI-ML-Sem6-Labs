from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^$', views.index, name='index'),
    re_path(r'^create/', views.create, name='create'),
    re_path(r'^createbook/', views.createbook, name='createbook'),
    re_path(r'^index', views.indexprod, name='add'),
    re_path(r'^addprod/', views.addprod, name='addprod'),
    re_path(r'^createprod/', views.createprod, name='createprod'),
    re_path(r'^addhuman/', views.addhuman, name='addhuman'),
    re_path(r'^createhuman/', views.createhuman, name='createhuman'),
    re_path(r'^update',views.update,name='update'),
]