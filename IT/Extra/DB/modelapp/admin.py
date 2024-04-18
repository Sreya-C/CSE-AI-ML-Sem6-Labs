from django.contrib import admin
from modelapp.models import BlogPost,CategoryModel,PageModel,EmpModel, InstituteModel

# Register your models here.
class BlogPostAdmin(admin.ModelAdmin):
    list_display = ('title', 'timestamp')
admin.site.register(BlogPost,BlogPostAdmin)

class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name','visits','likes')
admin.site.register(CategoryModel,CategoryAdmin)

class PageAdmin(admin.ModelAdmin):
    list_display = ('category','title','views')
admin.site.register(PageModel,PageAdmin)

class EmpAdmin(admin.ModelAdmin):
    list_display = ('pname','cname','salary','street','city')
admin.site.register(EmpModel,EmpAdmin)

class InstituteAdmin(admin.ModelAdmin):
    list_display = ('iid','name','nocourse')
admin.site.register(InstituteModel,InstituteAdmin)