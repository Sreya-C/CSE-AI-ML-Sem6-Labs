from django.urls import path
from . import views
urlpatterns = [   
    path('archive/', views.archive, name="archive"),
    path('archive/createblog/', views.create_blog, name="createblog"),
    path('', views.index, name='index'),
    path('createpage/', views.create, name='create'),
    path('home/add/', views.add, name='add'),
    path('home/get/', views.get, name='filter'),
    path('home/',views.home,name='home'),
]  