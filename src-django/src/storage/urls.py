"""
Media storage url patterns.
"""

from django.urls import path
from . import views


urlpatterns = [
    path(
        "images/download/<str:pathname>/",
        view=views.ImageDownloadView.as_view(),
        name="Image Download Endpoint",
    ),
    path(
        "images/reference/<str:pathname>/",
        view=views.ImageReferenceView.as_view(),
        name="Image URL Reference Viewpoint",
    ),
]
