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
import requests
from helpers import encode_jwt_token

ak = "7b5b3413fac847e79f4bdaf3d3d8fdc0" # fill access key kling
sk = "a159a6177fc745eca99f719c9f36941f" # fill secret key kling


class EEGModelReadView(APIView):

    def get(self, request: Request, eeg_id: int) -> Response:

        eeg_model_klass = get_object_or_404(models.EEGModel, id=eeg_id)
        eeg_model_serializer = serializers.EEGModelSerializer(eeg_model_klass)
        if eeg_model_serializer.is_valid():
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(data=eeg_model_serializer.data, status=status.HTTP_200_OK)


class EEGModelCreateView(APIView):

    def post(self, request: Request) -> Response:

        eeg_model_serializer = serializers.EEGModelSerializer(data=request.data)
        if eeg_model_serializer.is_valid():
            eeg_model_serializer.save()
            return Response(status=status.HTTP_201_CREATED)

        return Response(status=status.HTTP_400_BAD_REQUEST)


class DiffuserGenerateVideoView(APIView):
    
    def __init__(self):
        authorization = encode_jwt_token(ak, sk)
        self.response_dict = {}
        self.task_id = -1
        self.headers = {'Content-Type': 'application/json',
                   'Authorization': authorization
                   }
        self.video_info = {}
        self.video_url = ''

    def post(self, request: Request) -> Response:
        """
        Sends a request to the kling api to start converting an image into a 5 second video.
        """
        url = 'https://api.klingai.com/v1/videos/image2video'
        # Example data for testing
        data = {'model_name': 'kling-v1',
                'image': 'https://i.ebayimg.com/images/g/FmgAAOSwWeZfhwGA/s-l1600.jpg'
            }
        # Send post request
        response = requests.post(url, headers=self.headers, json=data)
        # Turn response into dictionary
        self.response_dict = response.json()
        try:
            # Extract the task_id from the response
            task_id = self.response_dict.get('data', {}).get('task_id', None)
            if task_id is None:
                print("Task ID not found in the response!")
            else:
                print(f"Task ID: {task_id}")
        except Exception as e:
            print(f"Error while extracting task_id: {e}")
        
        
        return Response(status=status.HTTP_201_CREATED)
    
    def get(self, request: Request, id: int) -> Response:
        """
        Sends a request to the kling api to retrieve the video that was generated.
        """
        # I'm not sure if this URL is correct
        url = f'https://api.klingai.com/v1/videos/image2video/{self.task_id}'
        # Example data for testing
        params = {'task_id': self.task_id}
        # Send get request
        response = requests.get(url, headers=self.headers, params=params)
        self.video_info = response.json()
        try:
            # Extract the task_id from the response
            self.video_url = self.video_info.get('data', {}).get('videos', None).get('url', '')
            if self.video_url is None:
                print("Video URL not found in the response!")
            else:
                print(f"Video URL: {self.video_url}")
        except Exception as e:
            print(f"Error while extracting video url: {e}")
        
        return Response(data=response.data, status=status.HTTP_200_OK)
