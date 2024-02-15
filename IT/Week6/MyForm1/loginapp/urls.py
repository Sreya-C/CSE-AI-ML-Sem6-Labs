from django.urls import re_path 
from . import views
urlpatterns = [ re_path(r'^$', views.login, name='login'),
				re_path(r'^$', views.login, name='loggedin'),
]