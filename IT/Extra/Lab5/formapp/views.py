from django.http import HttpRequest
from django.shortcuts import render, redirect
from . forms import RegForm, LoginForm, CarForm, FirstForm,RegistrationForm,MarksForm

def home(request):
    context = {}
    form = RegForm()
    context['form'] = form
    print(form.errors)
    return render(request,'home.html',{'form':form})


def login(request):
    username = "not logged in"
    cn = "not found"
    if request.method=="POST":
        MyLoginForm = LoginForm(request.POST)
        if MyLoginForm.is_valid():
            username = MyLoginForm.cleaned_data["username"]
            cn = MyLoginForm.cleaned_data["contact"]
            context = {"username":username,"contact":cn}
            return render(request,'loggedin.html',context)

    MyLoginForm = LoginForm()
    return render(request,'login.html',{'form':MyLoginForm})


def sesslogin(request):
    if request.method == 'POST':
        SessLoginForm = LoginForm(request.POST)
        if SessLoginForm.is_valid():
            username = SessLoginForm.cleaned_data["username"]
            request.session["username"] = username
            return render(request,'loggedin.html',{'username':username})
    
    SessLoginForm = LoginForm()
    return render(request,'login.html',{'form':SessLoginForm})

def formView(request):
    if request.session.has_key('username'):
        username = request.session['username']
        return render(request,'loggedin.html',{"username":username})
    else:
        return render(request,'login.html',{})

# def logout(request):
#     try:
#         del request.session['username']
#     except:
#         pass
#     return HTTPResponse("<strong>You are logged out</strong>")

def carhome(request):
    if request.method == "POST":
        form = CarForm(request.POST)
        if form.is_valid():
            manufac = form.cleaned_data["manufac"]
            model = form.cleaned_data["model"]
            context = {"manufac":manufac,"model":model}
            return render(request,'second.html',context)
    form = CarForm()
    return render(request,'carhome.html',{'form':form})

def firstpage(request:HttpRequest):
    if request.method == 'POST':
        form = FirstForm(request.POST)
        print(form.errors)
        print(form.is_valid())
        if form.is_valid():
            name = form.cleaned_data["name"]
            roll = form.cleaned_data["roll"]
            subject = form.cleaned_data["subject"]
            request.session["name"] = name
            request.session["roll"] = roll
            request.session["subject"] = subject
            context = {"name":name,"roll":roll,"subject":subject}
            return render(request,"second.html",context)
    form = FirstForm()
    return render(request,"firstpage.html",{"form":form})

def firstView(request):
    if request.session.has_key('name'):
        name = request.session["name"]
        roll = request.session["roll"]
        subject = request.session["subject"]
        return render(request,"second.html",{"name":name,"roll":roll,"subject":subject})
    else:
        return render(request,"firstpage.html",{})
    
def register(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            email = form.cleaned_data["email"]
            contact = form.cleaned_data["contact"]
            context = {"username": username,"email":email,"contact":contact}
            return render(request,"success.html",context)
    form = RegistrationForm()
    return render(request,"register.html",{"form":form})

def marks(request:HttpRequest):
    if request.method == "POST":
        form = MarksForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data["name"]
            marks = int(form.cleaned_data["marks"])/50.0
            request.session["name"] = name
            request.session["marks"] = marks
            context = {"marks":marks,"name":name}
            return render(request,"display.html",context)
    form = MarksForm()
    return render(request,"input.html",{"form":form})

def marksView(request):
    if request.session.has_key("marks") and request.session.has_key("name"):
        context = {"name":request.session["name"], "marks":request.session["marks"]}
        return render(request,"display.html",context)
    else:
        return render(request,"display.html",{})

