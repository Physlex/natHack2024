"""
Models defined for the api endpoints.
"""

from django.db import models


class EEGSample(models.Model):
    has_event = models.BooleanField()  # Determines if the sample is an 'event' type.
    frame = models.ForeignKey("EEGFrame", on_delete=models.CASCADE)
    data = models.FloatField()


class EEGFrame(models.Model):
    sample_rate = models.FloatField()
    model = models.ForeignKey("EEGModel", on_delete=models.CASCADE)


class EEGModel(models.Model):
    name = models.TextField()
