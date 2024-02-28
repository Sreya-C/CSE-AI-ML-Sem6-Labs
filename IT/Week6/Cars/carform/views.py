from django.shortcuts import render,redirect
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from .forms import CarForm

def home(request: HttpRequest):
    form = CarForm()
    return render(request, 'home.html', {'form':form})


def newpage(request):
    if request.method == 'POST':
        form = CarForm(request.POST)
        if form.is_valid():
            context = {
                'manufacturer': form.cleaned_data['manufacturer'],
                'model_name': form.cleaned_data['model_name'],
            }
            return render(request, 'newpage.html', context)
    return HttpResponse("Invalid Parameters")