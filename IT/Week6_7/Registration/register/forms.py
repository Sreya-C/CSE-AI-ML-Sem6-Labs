from django import forms   

class RegisterForm(forms.Form):
    username = forms.CharField(label="Username",required=True)
    password = forms.CharField(widget=forms.PasswordInput)
    email = forms.EmailField()
    phone = forms.IntegerField()
