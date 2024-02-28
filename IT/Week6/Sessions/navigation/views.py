from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import FirstForm

def firstpage(request):
    if request.method == 'POST':
        form = FirstForm(request.POST)
        if form.is_valid():
            request.session['name'] = form.cleaned_data['name']
            request.session['roll'] = form.cleaned_data['roll']
            request.session['subjects'] = form.cleaned_data['subjects']
            return redirect('secondpage')
    else:
        form = FirstForm()
    return render(request, 'firstpage.html', {'form': form})
        
def secondpage(request):
    name = request.GET.get('name')
    roll = request.GET.get('roll')
    subjects = request.GET.get('subjects')
    if not name or not roll or not subjects:
        return HttpResponse("Data not found in session")
    return render(request, 'secondpage.html', {'name': name, 'roll': roll, 'subjects': subjects})
    


    
            