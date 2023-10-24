# Users/urls.py

from django.urls import path
from knox.views import LogoutView
from knox.views import LogoutAllView
from . import views
from django.views.decorators.csrf import csrf_exempt

app_name = 'users'
urlpatterns = [
    path('api-login/', views.LoginView.as_view()),
    path('api-logout/', LogoutView.as_view()),
    path('api-logout-all/', LogoutAllView.as_view()),
    path('profile/', views.profile),
    path('add_ot_token/<str:token>', views.add_ot_token),
    path('stripe-checkout/', csrf_exempt(views.CreateCheckOutSession.as_view()), name='checkout_session'),
    path('webhook/', views.my_webhook_view),
    path('token_price/', views.token_price)
]
