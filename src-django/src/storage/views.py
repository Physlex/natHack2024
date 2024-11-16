"""
Views for downloading from media storage.
"""

from rest_framework.request import Request
from rest_framework.response import Response
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework import status

from . import models
from . import serializers


class ImageDownloadView(APIView):
    """
    Returns an image as a file for a user to download.
    """

    def get(request: Request, pathname: str) -> FileResponse:
        imageKlass = get_object_or_404(models.Image, url=pathname)
        image = open(imageKlass.url.path, "rb")
        return FileResponse(image, as_attachment=True)


class ImageReferenceView(APIView):
    """
    Returns an image url we can use as a reference to render.
    """

    def get(request: Request, pathname: str) -> Response:
        imageKlass = get_object_or_404(models.Image, url=pathname)
        imageData = serializers.ImageSerializer(imageKlass)
        if imageData.is_valid():
            return Response(data=imageData, status=status.HTTP_200_OK)
        else:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)
