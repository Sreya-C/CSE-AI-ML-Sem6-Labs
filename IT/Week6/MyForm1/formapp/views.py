from django.shortcuts import render
from .forms import RegForm,GeeksForm
def home_view(request):
	context = {}
	form = GeeksForm(request.POST or None)
	context['form'] = form
	return render(request, "home.html", context)
