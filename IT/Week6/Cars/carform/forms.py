from django import forms   

class CarForm(forms.Form):
    choices = [
        ('Suzuki', 'Suzuki'),
        ('Toyota', 'Toyota'),
        ('Honda', 'Honda'),
    ]
    manufacturer = forms.ChoiceField(label="Car Manufacturer", choices=choices)
    model_name = forms.CharField(label="Model Name")