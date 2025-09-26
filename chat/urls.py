from django.urls import path
from . import views

urlpatterns = [
    path('health', views.health, name='health'),
    path('models', views.list_models, name='models'),
    path('chat', views.chat_completion, name='chat'),
    path('chat/stream', views.chat_stream, name='chat_stream'),
    path('conversations', views.list_conversations, name='list_conversations'),
    path('conversations/<uuid:conv_id>', views.get_conversation, name='get_conversation'),
    path('conversations/<uuid:conv_id>/clear', views.clear_conversation, name='clear_conversation'),
]
