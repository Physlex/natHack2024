"""
Media storage url patterns.
"""

from django.urls import path
from . import views


urlpatterns = [
    path(
        "images/download/<int:image_id>/",
        view=views.ImageDownloadView.as_view(),
        name="Image Download Endpoint",
    ),
    path(
        "images/reference/<int:image_id>/",
        view=views.ImageReferenceView.as_view(),
        name="Image URL Reference Viewpoint",
    ),
]
