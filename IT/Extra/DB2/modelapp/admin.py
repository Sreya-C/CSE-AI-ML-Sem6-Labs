from django.contrib import admin
from modelapp.models import BookModel,PublisherModel,AuthorModel,ProductModel, HumanForm, HumanModel,StudentModel,StudentForm

# Register your models here.
class AuthorAdmin(admin.ModelAdmin):
    list_display = ('fname','lname','email')
admin.site.register(AuthorModel,AuthorAdmin)

class PublisherAdmin(admin.ModelAdmin):
    list_display = ('name','city')
admin.site.register(PublisherModel,PublisherAdmin)

class BookAdmin(admin.ModelAdmin):
    list_display = ('title','pubdate')
admin.site.register(BookModel,BookAdmin)

class ProductAdmin(admin.ModelAdmin):
    list_display = ('title','price')
admin.site.register(ProductModel,ProductAdmin)

class HumanAdmin(admin.ModelAdmin):
    list_display = ('pid','name','phone','addr')
admin.site.register(HumanModel,HumanAdmin)

class StudentAdmin(admin.ModelAdmin):
    list_display = ('sid','name','cname','dob')
admin.site.register(StudentModel,StudentAdmin)
