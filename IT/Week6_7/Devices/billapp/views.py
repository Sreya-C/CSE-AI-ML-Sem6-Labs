from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import DeviceForm  # Fixed import here

def selectpage(request):
    if request.method == 'POST':
        form = DeviceForm(request.POST)
        if form.is_valid():  # Ensure form data is valid
            request.session['company'] = form.cleaned_data['company']
            request.session['product'] = form.cleaned_data['product']
            request.session['quant'] = form.cleaned_data['quant']
            return redirect('billpage')
    else:
        form = DeviceForm()
    return render(request, 'home.html', {'form': form})  

def billpage(request):
    company = request.session.get('company')
    products = request.session.get('product')  # This is a list
    quant = request.session.get('quant')
    
    # Initial value based on company
    company_prices = {'HP': 5000, 'Nokia': 1000, 'Samsung': 8000, 'Motorola': 3000, 'Apple': 10000}
    val = company_prices.get(company, 0)  # Default to 0 if company not found
    
    # Adjust price based on product type
    if 'Mobile' in products:
        val += 50000
    if 'Laptop' in products:
        val += 100000
    
    finalamt = val * int(quant)
    
    if not company or not products or not quant:
        return HttpResponse("Data not found in session")
    
    return render(request, 'billpage.html', {'company':company,'quant':quant,'finalamt':finalamt})
