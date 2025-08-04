# yoga/urls.py (inside your app folder)
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.signup, name='signup'),
    path('login/', views.login_user, name='login'),
    path('upload/', views.upload_pose, name='upload'),
    path('result/', views.result, name='result'),
]
