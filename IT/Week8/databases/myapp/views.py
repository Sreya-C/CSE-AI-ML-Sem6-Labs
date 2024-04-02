from django.shortcuts import render
from datetime import datetime 
from django.http import HttpResponseRedirect 
from .models import BlogPost, BlogPostForm ,  CategoryModel, PageModel, EmpModel
from .forms import CategoryForm, PageForm, EmpForm

def archive(request):  
    posts = BlogPost.objects.all()[:10]     
    return render(request, 'archive.html',{'posts': posts, 'form': BlogPostForm()})  

def create_blog(request):     
    if request.method == 'POST':  
        form = BlogPostForm(request.POST)  
        if form.is_valid():  
            post = form.save(commit=False)  
            post.timestamp=datetime.now()  
            post.save()  
        return HttpResponseRedirect('/archive/') 

def index(req):
    categories = CategoryModel.objects.all()
    pages = PageModel.objects.all()
    return render(req, 'index.html', {'categories': categories, 'pages': pages, 'cat_form': CategoryForm(), 'page_form': PageForm()})

def create(req):
    if req.method == 'POST':
        cat_form = CategoryForm(req.POST)
        page_form = PageForm(req.POST)
        if cat_form.is_valid() and page_form.is_valid():
            cat = cat_form.save(commit=False)
            page = page_form.save(commit=False)
            if CategoryModel.objects.last():
                cat.index = CategoryModel.objects.last().index + 1
                page.index = PageModel.objects.last().index + 1
            else:
                cat.index = 1
                page.index = 1
            cat.save()
            page.save()
    return HttpResponseRedirect('/')


def home(req):
    emps = EmpModel.objects.all()
    return render(req, 'emp.html', {'emps': emps, 'form': EmpForm()})

def add(req):
    if req.method == 'POST':
        form = EmpForm(req.POST)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.save()
    return HttpResponseRedirect('/home/')

def get(req):
    if req.method == 'POST':
        company = req.POST['company']
   
        if company:
            emps = EmpModel.objects.filter(company=company)
            return render(req, 'filter.html', {'emps': emps})

    return HttpResponse("<h1>Incorrect Response</h1>")