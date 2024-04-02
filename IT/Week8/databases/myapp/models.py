from django.db import models
from django import forms

class BlogPost(models.Model):     
    title = models.CharField(max_length = 150) 
    body = models.TextField()     
    timestamp = models.DateTimeField()  
    class Meta:  
        ordering = ('-timestamp',)  

class BlogPostForm(forms.ModelForm): 
    class Meta:  
        model = BlogPost         
        exclude = ('timestamp',) 


class CategoryModel(models.Model):
    index = models.PositiveIntegerField()
    name = models.CharField(max_length=100)
    visits = models.PositiveIntegerField()
    likes = models.PositiveIntegerField()
    class Meta:
        ordering = ("index",)

class PageModel(models.Model):
    index = models.PositiveIntegerField()
    category = models.CharField(max_length=100)
    title = models.CharField(max_length=100)
    url = models.URLField()
    views = models.PositiveIntegerField()
    class Meta:
        ordering = ("index",)


class EmpModel(models.Model):
    name = models.CharField(max_length=100, primary_key=True)
    company = models.CharField(max_length=30)
    salary = models.PositiveIntegerField()
    street = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    class Meta:
        ordering = ('name',)