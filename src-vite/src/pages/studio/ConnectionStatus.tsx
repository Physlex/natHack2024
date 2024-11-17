/**
 * Connection Status Indictator component
 */


import { Box, Divider, Stack, Typography } from '@mui/material';
import React, { useEffect, useState } from 'react';


type ConnectionSidebarParams = {
    connectionStatus: string;
    deviceId: string;
    port: number;
    serialPort: string;
    startTime: Date;
    children: React.ReactNode;
}

type ConnectionSidebarProps = {
    interval: number;
}

export default function ConnectionSidebar({
    connectionStatus,
    port,
    serialPort,
    deviceId,
    startTime,
    children,
}: ConnectionSidebarParams): JSX.Element {
    const [connectionSidebarState, setConnectionSidebarState] = useState({
        interval: 0
    } as ConnectionSidebarProps);

    useEffect(() => {
        const now = new Date();
        const diffMs = (now.getTime() - startTime.getTime()) / 1000;
        setConnectionSidebarState({
            ...connectionSidebarState,
            interval: diffMs
        });
    }, [startTime])

    return (
        <Stack
            className="connection-status"
            spacing={3}
            divider={<Divider />}
            height="80vh">
            <Box>
                <Typography variant="h6">Connection Status</Typography>
            </Box>
            <Box className="connection-window" sx={{
                flexGrow: 1,
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between"
            }}>
                <Stack spacing={2}>
                    <Typography fontFamily={"monospace"}>
                        Port: {port}
                    </Typography>
                    <Typography fontFamily={"monospace"}>
                        Status: {connectionStatus}
                    </Typography>
                    <Typography fontFamily={"monospace"}>
                        Device ID: {deviceId}
                    </Typography>
                    <Typography fontFamily={"monospace"}>
                        Serial Port: {serialPort}
                    </Typography>
                    <Typography fontFamily={"monospace"}>
                        Time Elapsed: {connectionSidebarState.interval}
                    </Typography>
                </Stack>
                <Box className="connection-input" sx={{marginTop: "auto"}}>
                    {children}
                </Box>
            </Box>
        </Stack>
    );
}
