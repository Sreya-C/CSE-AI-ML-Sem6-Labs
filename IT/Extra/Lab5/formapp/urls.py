from django.urls import re_path
from . import views

urlpatterns = [
    #re_path(r'^home/',views.home,name='home'),
    #re_path(r'^login/',views.login,name='login')
    #re_path(r'^connection/',views.formView,name='formView'),
    #re_path(r'^login/',views.sesslogin,name='login'),
    #re_path(r'^logout/',views.logout,name='logout'),
    #re_path(r'^carhome/',views.carhome,name='carhome'),
    #re_path(r'^firstView/',views.firstView,name='firstView'),
    #re_path(r'^firstpage/',views.firstpage,name='firstpage'),
    #re_path(r'^register/',views.register,name='register'),
    re_path(r'^input/',views.marks,name='marks'),
    re_path(r'^inputView/',views.marksView,name='marksView'),   
]

