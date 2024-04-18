from django.shortcuts import render,get_object_or_404, redirect
from datetime import datetime
from django.http import HttpResponseRedirect, HttpResponse
from .models import BlogPost, BlogPostForm, CategoryModel, PageModel, CategoryForm, PageForm, EmpModel, EmpForm, InstituteModel, InstituteForm

# Create your views here.
def archive(request):
    print("Entered archive")
    posts = BlogPost.objects.all()[:10]
    return render(request, 'archive.html', {"posts": posts, "form": BlogPostForm()})

def create_blogpost(request):
    print("Entered")
    if request.method == "POST":
        form = BlogPostForm(request.POST)
        print(form.errors)
        print(form.is_valid())
        if form.is_valid():
            post = form.save(commit=False)
            post.timestamp = datetime.now()
            post.save()
    return HttpResponseRedirect('/')

def index(request):
    print("Entered index")
    categories = CategoryModel.objects.all()
    pages = PageModel.objects.all()
    return render(request,'index.html',{'categories':categories,"pages":pages,"cat_form":CategoryForm(),"page_form":PageForm()})

def createpage(request):
    print("Entered Create")
    if request.method == 'POST':
        cat_form = CategoryForm(request.POST)
        page_form = PageForm(request.POST)
        print(cat_form.errors)
        print(page_form.errors)
        if cat_form.is_valid() and page_form.is_valid():
            cat = cat_form.save(commit=False)
            page = page_form.save(commit=False)
            cat.save()
            page.save()
    return HttpResponseRedirect('/')

def home(request):
    print("Entered employeedata")
    employees = EmpModel.objects.all()
    return render(request,'emp.html',{'employees':employees,"emp_form":EmpForm()})

def add(request):
    if request.method == 'POST':
        form = EmpForm(request.POST)
        if form.is_valid():
            emp = form.save(commit=False)
            emp.save()
    return HttpResponseRedirect('/')

def get(request):
    if request.method == 'POST':
        cname = request.POST['cname']
        if cname:
            emps = EmpModel.objects.filter(cname=cname)
            return render(request,'filter.html',{'emps':emps,"company":cname})
    return HttpResponse("<h1> Incorrect Response</h1>")

def institute(request):
    institutes = InstituteModel.objects.all()
    return render(request,'institutes.html',{'institutes':institutes,'ins_form':InstituteForm()})

def ihome(request):
    if request.method == 'POST':
       form = InstituteForm(request.POST)
       print(form.errors)
       if form.is_valid():
           ins = form.save(commit=False)
           ins.save()
    return HttpResponseRedirect('/') 
           
def edit_institute(request, pk):
    instance = get_object_or_404(InstituteModel, pk=pk)
    form = InstituteForm(request.POST, instance=instance)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('institute')
    return render(request, 'edit_institute.html', {'form': form})

def delete_institute(request, pk):
    instance = get_object_or_404(InstituteModel, pk=pk)
    instance.delete()
    return redirect('institute')