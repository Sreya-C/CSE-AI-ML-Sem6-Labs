from django.http import HttpResponseRedirect
from django.shortcuts import render
from . import models, forms

# Create your views here.
#Q1
def index(req):
    authors = models.AuthorModel.objects.all()
    publishers = models.PublisherModel.objects.all()
    books = models.BookModel.objects.all()

    return render(req, 'book.html', {'authors': authors, 'author_form': forms.AuthorForm(), 'publishers': publishers, 
    'publisher_form': forms.PublisherForm(), 'books': books, 'book_form': forms.BookForm()})

def create(req):
    if req.method == 'POST':
        author_form = forms.AuthorForm(req.POST)
        publisher_form = forms.PublisherForm(req.POST)
        if author_form.is_valid() and publisher_form.is_valid():
            author = author_form.save(commit=False)
            publisher = publisher_form.save(commit=False)
            author.save()
            publisher.save()
        else:
            print("ERROR")
            print("Authors Valid : ", author_form.is_valid())
            print("Publisher Valid : ", publisher_form.is_valid())

        if not author_form.is_valid():
            return render(req, 'err.html', {'form': author_form})
        elif not publisher_form.is_valid():
            return render(req, 'err.html', {'form': publisher_form})

    return HttpResponseRedirect('/')

def createbook(req):
    if req.method == 'POST':
        book_form = forms.BookForm(req.POST)
        if book_form.is_valid():
            book = book_form.save(commit=False)
            book.save()
        else:
            print("ERROR")
            print("Book Valid : ", book_form.is_valid())
        if not book_form.is_valid():
            return render(req, 'err.html', {'form': book_form})

    return HttpResponseRedirect('/')

#Q2
def indexprod(req):
    prods = models.ProductModel.objects.all()
    return render(req, 'prod.html', {'prods': prods})

def addprod(req):
    return render(req, 'add.html', {'prod_form': forms.ProductForm()})

def createprod(req):
    if req.method == 'POST':
        prod_form = forms.ProductForm(req.POST)
        if prod_form.is_valid():
            prod = prod_form.save(commit=False)
            prod.save()
        else:
            print("ERROR")
    return HttpResponseRedirect('/index')

#Q3
def addhuman(req):
    return render(req, 'addhuman.html', {'hum_form': forms.HumanForm()})

def createhuman(req):
    if req.method == 'POST':
        hum_form = forms.HumanForm(req.POST)
        if hum_form.is_valid():
            hum = hum_form.save(commit=False)
            hum.save()
        else:
            print("ERROR")
    return HttpResponseRedirect('/update')

def update(req):
    hums = models.HumanModel.objects.all()
    return render(req, 'update.html', {'humans': hums})