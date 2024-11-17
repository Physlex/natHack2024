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
        timeseries = request.data["timeseries"]
        timestamps = request.data["timestamps"]
        if timeseries is None or timestamps is None:
            return Response(status=status.HTTP_400_BAD_REQUEST)

        eeg_model = models.EEGModel.objects.create()
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
                'image': 'https://cdn.discordapp.com/attachments/778329674215587841/1307550163048333353/IMG_20241116_203703.jpg?ex=673ab67b&is=673964fb&hm=b1b9461875a49e9e83256f2993e83509346c4b9ff55a578c648151ec7d1e4a48&',
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
                print(f"Task ID successfully extracted: {self.task_id}")
                
        except Exception as e:
            print(f"Error while extracting task_id: {e}")
            return Response(status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        
        data = {'task_id': self.task_id}
        return Response(status=status.HTTP_201_CREATED, data=data)
    
    def get(self, request: Request) -> Response:
        """
        Sends a request to the kling api to retrieve the video that was generated.
        """
        # Current task id: Cji7cGctxFMAAAAAAYQZmQ
        self.authorization = encode_jwt_token(self.ak, self.sk)
        print(f'auth: {self.authorization}')
        self.headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {self.authorization}'
                   }
        
        task_id = request.session.get('task_id', None)  # Retrieve task_id from the session
        # I'm not sure if this URL is correct
        url = f'https://api.klingai.com/v1/videos/image2video/Cji7cGctxFMAAAAAAYaNkg'
        # url = f'https://api.klingai.com/v1/videos/image2video/{task_id}'
        print(f'url: {url}')
        # Example data for testing
        params = {'task_id': self.task_id}
        # Send get request
        response = requests.get(url, headers=self.headers, params=params)
        # Debug
        print(f"GET Response Status Code: {response.status_code}")
        print(f"GET Response Content: {response.text}")
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
            return Response(data=self.video_info['data'], status=status.HTTP_418_IM_A_TEAPOT)
        
        return Response(data=self.video_info['data'], status=status.HTTP_200_OK)
    

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
