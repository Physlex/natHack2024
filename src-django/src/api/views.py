"""
API Endpoints are defined here.
"""


from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from django.http import FileResponse
from rest_framework.request import Request
from rest_framework.views import APIView
from . import serializers
from . import models


class EEGModelReadView(APIView):

    def get(request: Request, id: int) -> Response:

        eeg_model_klass = get_object_or_404(models.EEGModel, id=id)
        eeg_model_serializer = serializers.EEGModelSerializer(eeg_model_klass)
        if (eeg_model_serializer.is_valid()):
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(data=eeg_model_serializer.data, status=status.HTTP_200_OK)

class EEGModelCreateView(APIView):

    def post(request: Request) -> Response:

        eeg_model_serializer = serializers.EEGModelSerializer(data=request.data)
        if (eeg_model_serializer.is_valid()):
            eeg_model_serializer.save()
            return Response(status=status.HTTP_201_CREATED)

        return Response(status=status.HTTP_400_BAD_REQUEST)

class DiffuserGenerateVideoView(APIView):

    def post(request: Request) -> Response:
        return Response(status=status.HTTP_201_CREATED)

class DiffuserDownloadVideoView(APIView):

    def get(request: Request) -> FileResponse:

        # TODO: GENERATE FILE NAME FROM YOUR POST API RESPONSE
        filename = ""
        response = FileResponse(open(f"/videos/{filename}", 'rb'), as_attachment=True)
        response['Content-Disposition'] = f'attachment; filename="{filename}"'

        return response
