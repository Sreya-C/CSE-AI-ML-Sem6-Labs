from django import forms
from . import models

#Q1
class AuthorForm(forms.ModelForm):
    class Meta:
        model = models.AuthorModel
        exclude = ('index',)

class PublisherForm(forms.ModelForm):
    class Meta:
        model = models.PublisherModel
        exclude = ('index',)


class BookForm(forms.ModelForm):
    class Meta:
        model = models.BookModel
        exclude = ()

#Q2
class ProductForm(forms.ModelForm):
    class Meta:
        model = models.ProductModel
        exclude = ('',)

#Q3
class HumanForm(forms.ModelForm):
    class Meta:
        model = models.HumanModel
        exclude = ('',)