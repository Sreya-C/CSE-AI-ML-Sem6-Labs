from django.db import models
from django import forms
from django.core.exceptions import ValidationError


# Create your models here.
class BlogPost(models.Model):
    title = models.CharField(max_length=150)
    body = models.TextField()
    timestamp = models.DateTimeField()
    class Meta:
        ordering = ('-timestamp',)
    
class BlogPostForm(forms.ModelForm):
    class Meta:
        model = BlogPost
        fields = ['title','body']

class CategoryModel(models.Model):
    name = models.CharField(max_length=30)
    visits = models.IntegerField()
    likes = models.IntegerField()
    class Meta:
        ordering = ('name',)
    
class PageModel(models.Model):
    category = models.CharField(max_length=100)
    title = models.CharField(max_length=100)
    url = models.URLField()
    views = models.IntegerField()
    class Meta:
        ordering = ('category','title',)
    
class CategoryForm(forms.ModelForm):
    class Meta:
        model = CategoryModel
        fields = ['name','visits','likes']

class PageForm(forms.ModelForm):
    class Meta:
        model = PageModel
        fields = ['category','title','url','views']

class EmpModel(models.Model):
    pname = models.CharField(max_length=30)
    cname = models.CharField(max_length=40)
    salary = models.PositiveIntegerField()
    street = models.CharField(max_length=50)
    city = models.CharField(max_length=40)
    class Meta:
        ordering = ('pname','cname')

class EmpForm(forms.ModelForm):
    class Meta:
        model = EmpModel
        fields = ['pname','cname','salary','street','city']


class InstituteModel(models.Model):
    iid = models.CharField(max_length=5)
    name = models.CharField(max_length=50)
    nocourse = models.PositiveIntegerField()
    class Meta:
        ordering = ('iid',)
    
class InstituteForm(forms.ModelForm):
    class Meta:
        model = InstituteModel
        fields = ['iid','name','nocourse']

    def clean_nocourse(self):
        nocourse = self.cleaned_data.get('nocourse')
        if nocourse <= 10:
            raise ValidationError("Number of courses must be greater than 10.")
        return nocourse

