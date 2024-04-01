from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import RegisterForm

def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            # Storing data in session
            request.session['username'] = form.cleaned_data['username']
            request.session['email'] = form.cleaned_data['email']
            request.session['phone'] = form.cleaned_data['phone']
            return redirect('success')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})

def success(request):
    # Retrieving data from session
    username = request.session.get('username')
    email = request.session.get('email')
    phone = request.session.get('phone')
    
    if not username:
        return HttpResponse("Missing Required Data")
    
    return render(request, 'success.html', {'username': username, 'email': email, 'phone': phone})
