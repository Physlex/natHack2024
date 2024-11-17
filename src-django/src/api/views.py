"""
API Endpoints are defined here.
"""

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.views import APIView
from . import serializers
from . import models


class EEGModelReadView(APIView):

    def get(self, request: Request, eeg_id: int) -> Response:

        eeg_model_klass = get_object_or_404(models.EEGModel, id=eeg_id)
        eeg_model_serializer = serializers.EEGModelSerializer(eeg_model_klass)
        return Response(data=eeg_model_serializer, status=status.HTTP_200_OK)


class EEGModelCreateView(APIView):

    def post(self, request: Request) -> Response:
        timeseries = request.data['timeseries']
        timestamps = request.data['timestamps']
        if (timeseries is None or timestamps is None):
            return Response(status=status.HTTP_400_BAD_REQUEST)
        
        eeg_model = models.EEGModel.objects.create()
        samples = []
        for channel_idx, channel in enumerate(timeseries):
            eeg_channel = models.EEGChannel.objects.create(
                model=eeg_model,
                channel=channel_idx
            )
            for sample_idx, sample in enumerate(channel):
                samples.append(models.EEGSample.objects.create(
                    data=sample,
                    timestamp=timestamps[sample_idx],
                    channel=eeg_channel,
                ))

        models.EEGSample.objects.bulk_create(samples)
        return Response(status=status.HTTP_201_CREATED)


class DiffuserGenerateVideoView(APIView):

    def post(self, request: Request) -> Response:
        return Response(status=status.HTTP_201_CREATED)
