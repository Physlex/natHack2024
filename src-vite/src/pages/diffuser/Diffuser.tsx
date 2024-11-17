/**
 * Diffuser module.
 */


import React, { useState } from "react";
import { Grid2 as Grid, Stack, TextField, Button, Paper, Box } from "@mui/material";
import DownloadIcon from '@mui/icons-material/Download';


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

    // Update the timestamp setting for the page
    const handleTimestampChange = async (setting: string) => {
        setDiffuserState({
            ...diffuserState,
            timestampSetting: setting
        })
    }

    // Call the backend to generate the diffusion
    const handleButtonClick = async (_: React.MouseEvent<HTMLButtonElement>) => {
        const response = await fetch("/api/diffuser/generate/", {method: "POST"});
        if (response.status >= 400) {
            console.error(`Failed to generate diffusion. Reason: ${response.status}`);
            window.history.back();
            return;
        } else {
            
        }
    }
    // Enable/disabling download button when it is available
    //const [isDownloadEnabled, setDownloadEnabled] = useState(false);
    const [isDownloadEnabled] = useState(false);
    // Temporary code so I do not forget how this works
    // const handleEnableDownload = () => {
    //   setDownloadEnabled(true); // Enable the button
    // };

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
              justifyContent: 'center', // Space buttons apart
              flexDirection: 'column', 
              padding: 5, // Add padding to avoid edges
              margin: 3,
            }}
          >
            <Box
                sx={{
                    width: 800,
                    height: 450,
                    borderRadius: 1,
                    bgcolor: 'black',
                    margin: 3
                }}
            >
            </Box>
            {/* Grid to put buttons left to right instead */}
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
