"""
URLPatterns for the application api.
"""

from django.urls import path
from . import views


urlpatterns = [
    # EEG
    path(
        "/eeg/<int:eeg_id>",
        views.EEGModelReadView.as_view(),
        name="EEG Download Endpoint",
    ),
    path("/eeg", views.EEGModelCreateView.as_view(), name="EEG Creation Endpoint"),
    # DIFFUSER
    path(
        "/diffuser/generate",
        views.DiffuserGenerateVideoView.as_view(),
        name="ML Endpoint",
    ),
]
