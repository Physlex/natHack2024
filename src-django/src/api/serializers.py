"""
Serializers defn for api models.
"""

from rest_framework import serializers
from . import models


class EEGSampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.EEGSample
        fields = "__all__"


class EEGChannelSerializers(serializers.ModelSerializer):
    samples = EEGSampleSerializer(many=True)

    class Meta:
        model = models.EEGChannel
        fields = "__all__"


class EEGModelSerializer(serializers.ModelSerializer):
    frames = EEGChannelSerializers(many=True)

    class Meta:
        model = models.EEGModel
        fields = "__all__"
