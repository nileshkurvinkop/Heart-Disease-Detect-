from django.urls import path
from prof import views
urlpatterns=[
  path('',views.home,name='home'),
  path('predict/',views.predict,name='predict'),
]