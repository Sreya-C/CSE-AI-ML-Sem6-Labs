from django.contrib import admin
from . import models 

class AuthorAdmin(admin.ModelAdmin):
    list_display = ('fname', 'lname', 'email')
admin.site.register(models.AuthorModel, AuthorAdmin)

class PublisherAdmin(admin.ModelAdmin):
    list_display = ('name', 'url',)
admin.site.register(models.PublisherModel, PublisherAdmin)

class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'authors', 'publisher')
admin.site.register(models.BookModel, BookAdmin)

class ProductAdmin(admin.ModelAdmin):
    list_display = ('title', 'price')
admin.site.register(models.ProductModel, ProductAdmin)

class HumanAdmin(admin.ModelAdmin):
    list_display = ('fname','lname')
admin.site.register(models.HumanModel, HumanAdmin)