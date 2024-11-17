/**
 * Youtube URL Form module.
 */


import { Stack, Button, TextField } from '@mui/material';
import React, { useState } from 'react';


/**
 * @param label The associated form label
 * @param onSubmit Event listener to allow parent components to use the url on submission.
 */
type URLFormParams = {
    label: string;
    onSubmit: (url?: string) => void;
}

/**
 * The properties associated with a yt url form state.
 */
type URLProps = {
    url: string;
}

/**
 * @returns The youtube url form.
 */
export default function URLForm({ label, onSubmit }: URLFormParams): JSX.Element {
    const [urlFormState, setUrlFormState] = useState({
        url: ""
    } as URLProps);

    // Forwards submission to the parent element, allowing the parent access to the child
    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        console.info("Submitted url: ", urlFormState.url);
        onSubmit(urlFormState.url);
    }

    // Handle change implementation
    const handleChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const data = event.target.value;
        setUrlFormState({...urlFormState, url: data});
    }

    return (
        <Stack
            component="form"
            onSubmit={handleSubmit}
            sx={{
                display: "flex",
                flexGrow: 1,
                width: "100%",
                justifyContent: "center",
                alignItems: "center"
            }}
            flexDirection="row">
            <TextField
                type="url"
                id="outlined-basic"
                label={label}
                value={urlFormState.url}
                onChange={handleChange}
                required={true}
                fullWidth={true}
                variant="outlined" />
            <Button id="start-session-button" type="submit">
                Start Session
            </Button>
        </Stack>
    );
}
