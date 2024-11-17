/**
 * Diffuser module.
 */


import React, { useState } from "react";
import { Grid2 as Grid, Stack, TextField, Button, } from "@mui/material";


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

    return (
        <Grid container spacing={2} id="diffuser" component="div">
            <Grid>
                <SettingsMenu>
                    <Setting
                        name="timestamps"
                        label="Timestamp"
                        onChange={handleTimestampChange} />
                </SettingsMenu>
            </Grid>
            <Grid>
                <Button
                    className="generate-diffusion"
                    onClick={handleButtonClick}>
                    Generate!
                </Button>
                <Button
                    className="donwnload-diffusion">
                    Download
                </Button>
            </Grid>
        </Grid>
    );
}
