from django.shortcuts import render, redirect
from loginapp.forms import LoginForm

def login(request):
	username = "not logged in"
	cn = "not found"
	if request.method == "POST":
		form = LoginForm(request.POST)
		if form.is_valid():
			username = form.cleaned_data["username"]
			cn = form.cleaned_data["contact_num"]
			return redirect('loggedin')
	else:
		form = LoginForm()
		context = {'username': username,'contact_num':cn}
	return render(request, 'login.html', {'form': form})

def loggedin(request):
	if request.method == "POST":
		pass
	return render(request, 'loggedin.html')
