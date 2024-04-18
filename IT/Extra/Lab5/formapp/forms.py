from django import forms

class RegForm(forms.Form):
    title = forms.CharField(max_length=100)
    description = forms.CharField(max_length=100)
    views = forms.IntegerField()
    available = forms.BooleanField()
    
class LoginForm(forms.Form):
    username = forms.CharField(max_length=30)
    password = forms.CharField(widget=forms.PasswordInput())

class CarForm(forms.Form):
    choices = [
        ('Suzuki', 'Suzuki'),
        ('Toyota', 'Toyota'),
        ('Honda', 'Honda'),
    ]
    manufac = forms.ChoiceField(widget=forms.Select,label = "Car Manufacturer",choices = choices)
    model = forms.CharField(max_length=100)

class FirstForm(forms.Form):
    choices = [('Physics','Physics'),('Maths','Maths'),('Chemistry','Chemistry')]
    name = forms.CharField(max_length=30)
    roll = forms.IntegerField()
    subject = forms.ChoiceField(label="Subject",choices=choices)

class RegistrationForm(forms.Form):
    username = forms.CharField(max_length=30,label="Username")
    password = forms.CharField(widget=forms.PasswordInput,required=False)
    email = forms.EmailField(label="Email ID",required=False)
    contact = forms.CharField(label="Contact ID",max_length=10,required=False)

class MarksForm(forms.Form):
    name = forms.CharField(label="Name",max_length=30)
    marks = forms.CharField(label="Total Marks",max_length=3)
    