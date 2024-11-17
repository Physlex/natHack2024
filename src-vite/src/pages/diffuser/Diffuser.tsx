/**
 * Diffuser module.
 */


import React, { useState } from "react";
import { Grid2 as Grid, Stack, TextField, Button, Paper, Box } from "@mui/material";
import DownloadIcon from '@mui/icons-material/Download';
//import CircularProgress from '@mui/material/CircularProgress';


type SettingProps = {
    setting: string;
}

type SettingParams = {
    name: string;
    label: string;
    onChange: (setting: string) => void;
}

function Setting({ name, label, onChange }: SettingParams): JSX.Element {
    const [settingState, setSettingState] = useState({
        setting: ""
    } as SettingProps);

    // Event hook for handling the setting being changed on type
    const handleChange = async (event: React.ChangeEvent<HTMLTextAreaElement>) => {
        setSettingState({
            ...settingState,
            setting: event.target.value
        })
        onChange(event.target.value);
    };

    return (
        <TextField
            type="settings-field"
            name={name}
            onChange={handleChange}
            value={settingState.setting}
            label={label}
            required>
        </TextField>
    );
}

type SettingsMenuParams = {
    children: React.ReactNode;
};

function SettingsMenu({ children }: SettingsMenuParams): JSX.Element {
    return (
        <Stack>
            {children}
        </Stack>        
    );
}

type DiffuserProps = {
    timestampSetting: string;
    videoGenerated: boolean;
}

/**
 * @return Diffuser page.
 */
export default function Diffuser(): JSX.Element {
    const [diffuserState, setDiffuserState] = useState({
        timestampSetting: "",
        videoGenerated: false,
    } as DiffuserProps);

    // Enable/disabling download button when it is available
    //const [isDownloadEnabled, setDownloadEnabled] = useState(false);
    const [isDownloadEnabled, setDownloadEnabled] = useState(false);

    // State to toggle between Box and Video
    const [isVideo, setIsVideo] = useState(false);

    const [videoUrl, setVideoUrl] = useState<string | undefined>(undefined);

    //const [loading, setLoading] = React.useState(false);


    // Update the timestamp setting for the page
    const handleTimestampChange = async (setting: string) => {
        setDiffuserState({
            ...diffuserState,
            timestampSetting: setting
        })
    }


    // Call the backend to generate the diffusion
    // Button to call backend and generate the diffusion video from multiple images.
    // Will concatenate the videos generated.
    const handleButtonClick = async (_: React.MouseEvent<HTMLButtonElement>) => {
        const response = await fetch("/api/diffuser/generate/", {method: "POST"});
        if (response.status >= 400) {
            console.error(`Failed to generate diffusion. Reason: ${response.status}`);
            window.history.back();
            return;
        } else { // Need to wait for video to be created and load.
            const responseData = await response.json();
            const taskId = responseData.task_id;
            
            if (taskId != null) {
                console.log(taskId);
                // Step 3: Start polling the server every 5 seconds to check for the video_url
                //setLoading((prevLoading) => !prevLoading);
                
                const pollInterval = setInterval(async () => {
                  const statusResponse = await fetch(`/api/diffuser/generate/`, { method: "GET" });
                
                  if (statusResponse.status === 200) {
                    const statusData = await statusResponse.json();
                    // Check if task_result and videos exist in the response
                    const videoUrl = statusData.url;
                    if (videoUrl) {
                        // Stop polling once the video_url is received
                        clearInterval(pollInterval);
                        // Proceed to handle the video URL, e.g., show the video
                        console.log('Video URL:', videoUrl);
                        // You can update state here to show the video or enable download
                        setIsVideo(true); // Assuming this state will render the video
                        setVideoUrl(videoUrl); // Set the video URL for your component
                        setDownloadEnabled(true); // Enable download button or other UI
                    } else {
                        console.error('No video URL found in the response.');
                    }
                  } else {
                    console.error(`Failed to fetch video status. Reason: ${statusResponse.status}`);
                  }
                }, 5000); // 5 seconds interval
            }
        }

        // TEMPORARY BELOW!!
        // setLoading((prevLoading) => !prevLoading);
        // setIsVideo(true); // Assuming this state will render the video
        // setVideoUrl("https://cdn.klingai.com/bs2/upload-kling-api/2154537099/image2video/CjNQtmctxFMAAAAAAYUaYA-0_raw_video_1.mp4"); // Set the video URL for your component
        // setDownloadEnabled(true); // Enable download button or other UI
    }

    const handleDownloadClick = async () => {
        try {
          const response = await fetch("/api/diffuser/download-video/", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ url: videoUrl }), // Example URL
          });
      
          if (response.status === 200) {
            // Handle the video download, e.g., by redirecting the user to the response URL
            const blob = await response.blob();
            const downloadUrl = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = 'mind-diffuser-creation.mp4';  // You can set the name based on the filename from the backend
            link.click();
          } else {
            console.error('Failed to download video:', await response.text());
          }
        } catch (error) {
          console.error('Error while requesting video download:', error);
        }
      };



    return (
        <Grid container sx={{ height: '100vh' }}>
        {/* Left Box */}
        <Grid size={3}>
          <Paper elevation={5}
            sx={{
              backgroundColor: 'lightblue',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <SettingsMenu>
                <Setting
                name="timestamps"
                label="Timestamp"
                onChange={handleTimestampChange} />
            </SettingsMenu>
          </Paper>
        </Grid>
  
        {/* Right Box */}
        <Grid size={9}>
          <Paper elevation={1}
            sx={{
              backgroundColor: 'white',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'top', // Space buttons apart
              flexDirection: 'column', 
              padding: 1, // Add padding to avoid edges
            }}
          >
            {/* Conditionally render Box or Video */}
            {isVideo ? (
            <Box
                sx={{
                width: 800,
                height: 450,
                borderRadius: 1,
                bgcolor: 'black',
                margin: 3,
                justifyContent: 'center',
                }}
            >
                <video
                style={{ width: '100%', height: '100%', borderRadius: '1rem' }}
                controls
                autoPlay
                >
                <source src={videoUrl} type="video/mp4" />
                Your browser does not support the video tag.
                </video>
            </Box>
            ) : (
            <Box
                sx={{
                width: 800,
                height: 450,
                borderRadius: 1,
                bgcolor: 'black',
                margin: 3,
                }}
            />
            )}
            {/* Box to put buttons left to right instead */}
            <Box
                sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center", // Align buttons vertically in the center
                width: 800, // Ensure the box spans the full width
                }}
            >
                <Button
                    className="generate-diffusion"
                    variant="outlined"
                    size="large"
                    onClick={handleButtonClick}
                    >
                    
                    Generate!
                </Button>
                <Button
                    className="donwnload-diffusion"
                    variant="outlined"
                    size="large"
                    onClick={handleDownloadClick}
                    endIcon={<DownloadIcon />}
                    disabled={!isDownloadEnabled} // Button is disabled unless state is true
                    >
                    Download
                </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    );
}
