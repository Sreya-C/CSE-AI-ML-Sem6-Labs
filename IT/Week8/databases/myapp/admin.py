from django.contrib import admin
from .models import BlogPost, CategoryModel, PageModel,EmpModel
from . import models 

class BlogPostAdmin(admin.ModelAdmin):  
    list_display = ('title', 'timestamp')  
admin.site.register(models.BlogPost,BlogPostAdmin)

class CategoryAdmin(admin.ModelAdmin):
    list_display = ('index', 'name',)
admin.site.register(CategoryModel, CategoryAdmin)

class PageAdmin(admin.ModelAdmin):
    list_display = ('index', 'category', 'title',)
admin.site.register(PageModel, PageAdmin)

class EmpAdmin(admin.ModelAdmin):
    list_display = ('name', 'company', 'salary',)
admin.site.register(EmpModel, EmpAdmin)