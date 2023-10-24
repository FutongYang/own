"""
URL configuration for EduardOnline project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import include
from . import topography_api

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('Users.urls')),
    path('api/download_map/', topography_api.download_map, name='download_map'),
    path('api/elevation_maps/', topography_api.list_elevation_maps, name='list_elevation_maps'),
    path('api/elevation_maps/<int:map_id>/', topography_api.retrieve_elevation_map, name='retrieve_elevation_map'),
    path('api/elevation_maps/<int:map_id>/delete/', topography_api.delete_elevation_map, name='delete_elevation_map'),
    path('api/generate_map/', topography_api.generate_map, name='generate_map'),
    path('api/relief_maps/', topography_api.list_relief_maps)
]


