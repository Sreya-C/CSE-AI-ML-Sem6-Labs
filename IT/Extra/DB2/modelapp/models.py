from django.db import models
from django import forms
from operator import mod

# Create your models here.
class AuthorModel(models.Model):
    fname = models.CharField(max_length=30)
    lname = models.CharField(max_length=30)
    email = models.EmailField()
    class Meta:
        ordering = ('fname','lname',)
        
    def __str__(self):
        return self.fname
    
class PublisherModel(models.Model):
    name = models.CharField(max_length=30)
    street = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    country = models.CharField(max_length=30)
    site = models.URLField(blank=True)
    class Meta:
        ordering = ('name',)

    def __str__(self):
        return self.name

class BookModel(models.Model):
    title = models.CharField(max_length=60)
    pubdate = models.DateField()
    publisher = models.ForeignKey(PublisherModel,on_delete=models.CASCADE)
    authors = models.ManyToManyField(AuthorModel)
    class Meta:
        ordering = ('title',)
    
    def __str__(self):
        return self.title

class AuthorForm(forms.ModelForm):
    class Meta:
        model = AuthorModel
        fields = ['fname','lname','email']

class PublisherForm(forms.ModelForm):
    class Meta:
        model = PublisherModel
        fields = ['name','street','city','state','country']
    
class BookForm(forms.ModelForm):
    class Meta:
        model = BookModel
        fields = ['title','pubdate','publisher','authors']

class ProductModel(models.Model):
    title = models.CharField(max_length=40)
    price = models.PositiveIntegerField()
    description = models.CharField(max_length=200)
    class Meta:
        ordering = ('title',)

class ProductForm(forms.ModelForm):
    class Meta:
        model = ProductModel
        fields = ['title','price','description']

class HumanModel(models.Model):
    pid = models.PositiveIntegerField()
    name = models.CharField(max_length=30)
    phone = models.CharField(max_length=10)
    addr = models.CharField(max_length=50)
    class Meta:
        ordering = ('pid',)

class HumanForm(forms.ModelForm):
    class Meta:
        model = HumanModel
        fields = ['pid','name','phone','addr']

class StudentModel(models.Model):
    sid = models.PositiveIntegerField()
    name = models.CharField(max_length=30)
    cname = models.CharField(max_length=50)
    dob = models.DateField()
    class Meta:
        ordering = ('sid',)

class StudentForm(forms.ModelForm):
    class Meta:
        model = StudentModel
        fields = ['sid','name','cname','dob']
        