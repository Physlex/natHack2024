"""
Serializers defn for api models.
"""

from rest_framework import serializers
from . import models


class EEGSampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.EEGSample
        fields = ["id", "has_event", "data"]


class EEGFrameSerializer(serializers.ModelSerializer):
    samples = EEGSampleSerializer(many=True)

    class Meta:
        model = models.EEGFrame
        fields = ["sample_rate"]


class EEGModelSerializer(serializers.ModelSerializer):
    frames = EEGFrameSerializer(many=True)

    class Meta:
        model = models.EEGModel
        fields = ["id", "name"]
