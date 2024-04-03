from django.db import models
from django import forms

# Create your models here.
class AuthorModel(models.Model):
    fname = models.CharField(max_length=50)
    lname = models.CharField(max_length=50)
    email = models.EmailField()

    class Meta:
        ordering = ("fname",)

class PublisherModel(models.Model):
    name = models.CharField(max_length=100)
    street = models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    country = models.CharField(max_length=100)
    url = models.URLField()
    class Meta:
        ordering = ("name",)

class BookModel(models.Model):
    title = models.CharField(max_length=100)
    date = models.DateField()
    authors = models.ForeignKey(AuthorModel, on_delete=models.CASCADE)
    publisher = models.OneToOneField(PublisherModel, on_delete = models.CASCADE)
    class Meta:
        ordering = ("title",) 

#Q2
class ProductModel(models.Model):
    title = models.CharField(max_length=100)
    price = models.PositiveIntegerField()
    des = models.CharField(max_length=1000)

#Q3
class HumanModel(models.Model):
    fname = models.CharField(max_length=100)
    lname = models.CharField(max_length=100) 
    phone = models.PositiveIntegerField() 
    addr = models.CharField(max_length=100) 
    city = models.CharField(max_length=100) 