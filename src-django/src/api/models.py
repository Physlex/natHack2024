"""
Models defined for the api endpoints.
"""

from django.db import models


class EEGSample(models.Model):
    data = models.FloatField()
    timestamp = models.DecimalField(max_digits=20, decimal_places=6)
    channel = models.ForeignKey("EEGChannel", on_delete=models.CASCADE)


class EEGChannel(models.Model):
    channel = models.IntegerField()
    model = models.ForeignKey("EEGModel", on_delete=models.CASCADE)


class EEGModel(models.Model):
    name = models.CharField(max_length=255)
    date = models.DateTimeField(auto_now_add=True)
