/**
 * Dashboard page.
 */


import { Box } from '@mui/material';


/**
 * @returns Dashboard component.
 */
export default function Dashboard(): JSX.Element {
    return (
        <Box
            id="dashboard"
            sx={{
                padding: "10px",
                margin: "10px auto",
                height: "100%",
                width: "80%"
            }}
            component="div">
        </Box>
    );
}
