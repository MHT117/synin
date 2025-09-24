from django.urls import path
from . import views

urlpatterns = [
    path('health', views.health, name='health'),
    path('models', views.list_models, name='models'),
    path('chat', views.chat_completion, name='chat'),
]
