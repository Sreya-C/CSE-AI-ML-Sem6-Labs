from django.shortcuts import render, redirect
from django.http import HttpResponse,HttpResponseRedirect
from modelapp.models import AuthorModel,AuthorForm,PublisherForm,PublisherModel,BookForm,BookModel,ProductForm,ProductModel,HumanForm,HumanModel,HumanModel,StudentModel,StudentForm

# Create your views here.
def home(request):
    authors = AuthorModel.objects.all()
    publishers = PublisherModel.objects.all()
    books = BookModel.objects.all()
    return render(request,'books.html',{'authors':authors,'auth_form':AuthorForm(),'books':books,'publishers':publishers,'book_form':BookForm(),'pub_form':PublisherForm()})
    
def create(request):
    if request.method == 'POST':
        auth_form = AuthorForm(request.POST)
        pub_form = PublisherForm(request.POST)
        book_form = BookForm(request.POST)
        print(book_form.errors)
        print(auth_form.errors)
        print(pub_form.errors)
        if book_form.is_valid():
            print("it is a valid book")
            auth = auth_form.save(commit=False)
            pub = pub_form.save(commit=False)
            book = book_form.save(commit=False)
            auth.save()
            book.save()
            pub.save()
    return HttpResponseRedirect('/')


def product(request):
    products = ProductModel.objects.all()
    return render(request,'product.html',{"products":products,"prod_form":ProductForm()})

def createprod(request):
    if request.method == 'POST':
        form = ProductForm(request.POST)
        if form.is_valid():
            print("it is a valid product")
            prod = form.save(commit=False)
            prod.save()
    return HttpResponseRedirect('/')

def hhome(request):
    humans = HumanModel.objects.all()
    return render(request,'names.html',{"humans":humans,"hum_form":HumanForm()})

def details(request,pid):
    human = HumanModel.objects.filter(pid=pid).first()
    name = human.name
    addr = human.addr
    phone = human.phone
    return render(request,'details.html',{"name":name,"addr":addr,"phone":phone,"pid":pid})

def shome(request):
    students = StudentModel.objects.all()
    return render(request,'student.html',{'students':students,'form':StudentForm()})

def student(request):
    if request.method == 'POST':
        form = StudentForm(request.POST)
        if form.is_valid():
            stu = form.save(commit=False)
            stu.save()
    return HttpResponseRedirect('/')

def edit_student(request,sid):
    instance = StudentModel.objects.get(sid=sid)
    form = StudentForm(request.POST,instance=instance)
    if form.is_valid():
        stu = form.save(commit=False)
        stu.save()
        return redirect('student')
    return render(request,"edit.html",{"form":form,"sid":sid})

def delete(request,sid): 
    instance = StudentModel.objects.get(sid=sid)
    instance.delete()
    return HttpResponseRedirect('/')