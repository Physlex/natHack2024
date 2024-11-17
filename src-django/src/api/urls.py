"""
URLPatterns for the application api.
"""

from django.urls import path
from . import views


urlpatterns = [
    # EEG
    path(
        "eeg/<int:eeg_id>/",
        views.EEGModelReadView.as_view(),
        name="EEG Read Endpoint",
    ),
    path(
        "eeg/",
        views.EEGModelCreateView.as_view(),
        name="EEG Create Endpoint"
    ),
    # DIFFUSER
    path(
        "diffuser/generate/",
        views.DiffuserGenerateVideoView.as_view(),
        name="ML Endpoint",
    ),
    # Downloading Videos
    path(
        "diffuser/download-video/",
        views.DownloadVideoView.as_view(),
        name="Download Video",
    ),
]
