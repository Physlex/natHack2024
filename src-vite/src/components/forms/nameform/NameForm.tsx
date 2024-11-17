/**
 * Name form module.
 */


import { Box, TextField } from '@mui/material';
import React, { useState } from 'react';


type EEGNameForm = {
    onChange: (name: string) => void;
}

type EEGNameProps = {
    name: string;
}

/**
 * @returns Name Form Component
 */
export default function EEGNameForm({ onChange }: EEGNameForm): JSX.Element {
    const [eegNameFormState, setEEGNameFormState] = useState({
        name: ""
    } as EEGNameProps);

    // Submission handler
    const handleSubmit = (event: React.FormEvent<HTMLElement>) => {
        event.preventDefault();
    }

    // Change handler
    const handleChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
        const data = event.target.value;
        setEEGNameFormState({
            ...eegNameFormState,
            name: data
        });

        onChange(data);
    }

    return (
        <Box component="form" onSubmit={handleSubmit}>
            <TextField
                required
                id="outlined-basic"
                label={"Name"}
                value={eegNameFormState.name}
                onChange={handleChange}
                fullWidth={true}
                variant="outlined"
             ></TextField>
        </Box>
    );
}
