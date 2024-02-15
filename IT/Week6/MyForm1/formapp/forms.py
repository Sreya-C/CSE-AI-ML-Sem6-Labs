from django import forms
class RegForm(forms.Form):
	title = forms.CharField()
	description = forms.CharField()
	views = forms.IntegerField()
	available = forms.BooleanField()

class GeeksForm(forms.Form):
	title = forms.CharField(widget = forms.Textarea)
	description = forms.CharField(widget = forms.CheckboxInput)
views = forms.IntegerField(widget = forms.TextInput)
available = forms.BooleanField(widget = forms.Textarea)