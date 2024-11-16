from django.db import models

class Image(models.Model):
    name = models.TextField()
    url = models.ImageField(upload_to='images/')

# class Video(models.Model): TODO: Figure out how to do this
