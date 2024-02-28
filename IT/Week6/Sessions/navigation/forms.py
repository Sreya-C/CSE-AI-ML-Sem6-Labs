from django import forms

class FirstForm(forms.Form):
    choices = [
        ('Math', 'Math'),
        ('English', 'English'),
        ('Physics', 'Physics'),
    ]
    name = forms.CharField(label="Name")
    roll = forms.CharField(label="Roll")
    subjects = forms.ChoiceField(label="Subject", choices=choices)   