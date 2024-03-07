from django import forms

class DeviceForm(forms.Form):
    company_choices = [
        ('HP', 'HP'),
        ('Nokia', 'Nokia'),
        ('Samsung', 'Samsung'),
        ('Motorola', 'Motorola'),
        ('Apple', 'Apple'),
    ]
    company = forms.ChoiceField(widget=forms.RadioSelect, label="Company", choices=company_choices)
    
    product_types = [
        ('Mobile', 'Mobile'),
        ('Laptop', 'Laptop'),
    ]
    product = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple, label="Product", choices=product_types)
    
    quant = forms.CharField(label="Quantity")
