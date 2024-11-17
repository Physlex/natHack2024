# Generated by Django 5.1.3 on 2024-11-17 00:24

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="EEGModel",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name="EEGFrame",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("sample_rate", models.FloatField()),
                (
                    "model",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="api.eegmodel"
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="EEGSample",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("has_event", models.BooleanField()),
                ("data", models.FloatField()),
                (
                    "frame",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="api.eegframe"
                    ),
                ),
            ],
        ),
    ]