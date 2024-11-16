"""
Serializers for storage media objects.
"""


from rest_framework import serializers
from . import models


class ImageSerializer(serializers.ModelSerializer):
    """
    Serializer for images.
    """

    class Meta:
        model = models.Image
        fields = ["id", "name", "url"]
