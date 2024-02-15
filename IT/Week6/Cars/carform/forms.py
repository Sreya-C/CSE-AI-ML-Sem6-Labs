from django import forms 
class LoginForm(forms.Form):
	CHOICES= (('1','Toyota'),('2','Honda'),('3','Tata'),('4','Ford'))
	manufact = forms.ChoiceField(widget=forms.Select, choices=CHOICES)
	model = forms.CharField(widget = forms.Textarea)
	