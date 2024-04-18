from django.urls import re_path, path
from modelapp.views import archive,create_blogpost 
from . import views

urlpatterns = [ 
    #path('', archive, name="archive"),
    #path('createblog/', create_blogpost, name="createblog"),
    #path('',views.index,name='index'),
    #path('createpage/', views.createpage, name='createpage'),
    # path('add/', views.add, name='add'),
    # path('get/', views.get, name='filter'),
    # path('',views.home,name='home'),
    path('',views.institute, name='institute'),
    path('home/',views.ihome, name='home'),
    path('edit/<int:pk>/', views.edit_institute, name='edit_institute'),
    path('delete/<int:pk>/', views.delete_institute, name='delete_institute'),
]
