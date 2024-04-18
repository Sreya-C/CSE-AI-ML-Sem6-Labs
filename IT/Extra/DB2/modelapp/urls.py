from django.urls import path,include
from . import views
urlpatterns = [
    #path("",views.home,name='home'),
    #path("create/",views.create,name='create'),
    # path("",views.product,name='product'),
    # path("createprod/",views.createprod,name='createprod'),
    # path("",views.hhome,name='hhome'),
    # path("details/<int:pid>/",views.details,name='details'),
    path("",views.shome,name='shome'),
    path("student/",views.student,name='student'),
    path("edit/<int:sid>/",views.edit_student,name='edit'),
    path("delete/<int:sid>/",views.delete,name='delete'),
]
