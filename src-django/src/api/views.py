"""
API Endpoints are defined here.
"""


import os
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.views import APIView
from . import serializers
from . import models
import requests
from .helpers import encode_jwt_token
from django.http import HttpResponse


class EEGModelReadView(APIView):

    def get(self, request: Request, eeg_id: int) -> Response:

        eeg_model_klass = get_object_or_404(models.EEGModel, id=eeg_id)
        eeg_model_serializer = serializers.EEGModelSerializer(eeg_model_klass)
        return Response(data=eeg_model_serializer, status=status.HTTP_200_OK)


class EEGModelCreateView(APIView):

    def post(self, request: Request) -> Response:
        name = request.data["name"]
        timeseries = request.data["timeseries"]
        timestamps = request.data["timestamps"]
        if timeseries is None or timestamps is None:
            return Response(status=status.HTTP_400_BAD_REQUEST)

        eeg_model = models.EEGModel.objects.create(name=name)
        samples = []
        for channel_idx, channel in enumerate(timeseries):
            eeg_channel = models.EEGChannel.objects.create(
                model=eeg_model, channel=channel_idx
            )
            for sample_idx, sample in enumerate(channel):
                samples.append(
                    models.EEGSample.objects.create(
                        data=sample,
                        timestamp=timestamps[sample_idx],
                        channel=eeg_channel,
                    )
                )

        models.EEGSample.objects.bulk_create(samples)
        return Response(status=status.HTTP_201_CREATED)


class DiffuserGenerateVideoView(APIView):
    
    def __init__(self):
        """
        This was a bad idea because it turns out that
        a new class is instantiated every time a new request is made
        so this all gets overwritten haha
        """
        self.ak = "7b5b3413fac847e79f4bdaf3d3d8fdc0" # fill access key kling
        self.sk = "a159a6177fc745eca99f719c9f36941f" # fill secret key kling
        self.response_dict = {}
        self.task_id = -1
        self.video_info = {}
        self.video_url = ''

    def post(self, request: Request) -> Response: # This should have image url passed into it for processing
        """
        Sends a request to the kling api to start converting an image into a 5 second video.
        """
        self.authorization = encode_jwt_token(self.ak, self.sk)
        print(f'auth: {self.authorization}')
        self.headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {self.authorization}'
                   }
        # TODO: Add back!! 
        url = 'https://api.klingai.com/v1/videos/image2video'
        # Example data for testing
        data = {'model_name': 'kling-v1',
                'image': 'https://thumbs.dreamstime.com/b/neon-background-wallpaper-futuristic-glowing-lights-cool-backgrounds-blue-white-green-image-generated-use-ai-276345013.jpg',
            }
        # Send post request
        response = requests.post(url, headers=self.headers, json=data)
        # Debug
        print(f"POST Response Status Code: {response.status_code}")
        print(f"POST Response Content: {response.text}")
        # Turn response into dictionary
        self.response_dict = response.json()
        try:
            # Extract the task_id from the response
            task_id = self.response_dict.get('data', {}).get('task_id', None)
            if not task_id:
                print(f"Task ID missing! Response: {self.response_dict}")
            else:
                request.session['task_id'] = task_id
                print(f"Task ID successfully extracted: {task_id}")
                
        except Exception as e:
            print(f"Error while extracting task_id: {e}")
            return Response(status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        
        data = {'task_id': task_id}
        return Response(status=status.HTTP_201_CREATED, data=data)
    
    def get(self, request: Request, task_id: str) -> Response:
        """
        Sends a request to the Kling API to retrieve the video that was generated for a given task_id.
        """
        # Authorization
        self.authorization = encode_jwt_token(self.ak, self.sk)
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.authorization}'
        }
        
        # Construct the URL using the task_id
        url = f'https://api.klingai.com/v1/videos/image2video/{task_id}'
        print(f'Requesting video info from URL: {url}')
        
        try:
            # Send GET request to the Kling API
            response = requests.get(url, headers=self.headers)
            print(f"GET Response Status Code: {response.status_code}")
            print(f"GET Response Content: {response.text}")
            
            if response.status_code == 200:
                video_info = response.json()
                video_url = video_info["data"]["task_result"]["videos"][0]["url"]
                return Response(data={"url": video_url}, status=status.HTTP_200_OK)
            else:
                return Response(
                    data={"error": "Failed to retrieve video info from Kling API."},
                    status=response.status_code
                )
        except KeyError:
            return Response(
                data={"error": "Unexpected response structure from Kling API."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            return Response(
                data={"error": f"An error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    

class DownloadVideoView(APIView):

    def post(self, request: Request) -> Response:
        
        # Extract the URL from the request body
        url = request.data.get('url')
        
        if not url:
            return Response({"error": "URL is required"}, status=400)
        
        # Attempt to download the video from the URL
        try:
            response = requests.get(url, stream=True)

            # Check if the URL is valid
            if response.status_code != 200:
                return Response({"error": "Failed to fetch the video from the provided URL"}, status=400)
            
            # Get the video filename from the URL
            filename = os.path.basename(url)

            # Return the video as a response
            return HttpResponse(
                response.iter_content(chunk_size=1024),
                content_type='video/mp4',  # Adjust the content type if it's a different video format
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"'
                }
            )
        except Exception as e:
            return Response({"error": str(e)}, status=500)
        
        return Response(status=status.HTTP_201_CREATED)

class DashboardAttentionView(APIView):
    def get(self, request: Request, eeg_model_name: str) -> Response:
        eeg_model_klass = get_object_or_404(models.EEGModel, name=eeg_model_name)
        eeg_channel_klasses = models.EEGChannel.objects.filter(model=eeg_model_klass)

        timeseries = []
        timesamples = []
        for eeg_channel_klass in eeg_channel_klasses:
            eeg_time_series = models.EEGSample.objects.filter(channel=eeg_channel_klass)
            for eeg_sample in eeg_time_series:
                timeseries.append(eeg_sample.data)
                timesamples.append(eeg_sample.timestamp)

        attention_series = []
        attention_timesamples = []

        ## TODO: Attention series algorithm here

        return Response(data={
            "attentionseries": attention_series,
            "timesamples": attention_timesamples
        }, status=status.HTTP_200_OK)

class DashboardEEGView(APIView):
    def get(self, request: Request, eeg_model_name: str) -> Response:
        eeg_model_klass = get_object_or_404(models.EEGModel, name=eeg_model_name)
        eeg_channel_klasses = models.EEGChannel.objects.filter(model=eeg_model_klass)

        timeseries = []
        timesamples = []
        for eeg_channel_klass in eeg_channel_klasses:
            eeg_time_series = models.EEGSample.objects.filter(channel=eeg_channel_klass)
            for eeg_sample in eeg_time_series:
                timeseries.append(eeg_sample.data)
                timesamples.append(eeg_sample.timestamp)

        return Response(
            {
                "timeseries": timeseries,
                "timesamples": timesamples
            },
            status=status.HTTP_200_OK
        )

class DashboardEEGListView(APIView):
    def get(self, request: Request) -> Response:
        eeg_model_klasses = models.EEGModel.objects.all()
        eeg_models = []
        for eeg_model_klass in eeg_model_klasses:
            eeg_channel_klasses = models.EEGChannel.objects.filter(model=eeg_model_klass)

            timeseries = []
            timesamples = []
            for eeg_channel_klass in eeg_channel_klasses:
                eeg_time_series = models.EEGSample.objects.filter(channel=eeg_channel_klass)
                for eeg_sample in eeg_time_series:
                    timeseries.append(eeg_sample.data)
                    timesamples.append(eeg_sample.timestamp)

            eeg_models.append({
                "name": eeg_model_klass.name,
                "timeseries": timeseries,
                "timesamples": timesamples
            })

        return Response(data=eeg_models, status=status.HTTP_200_OK)
