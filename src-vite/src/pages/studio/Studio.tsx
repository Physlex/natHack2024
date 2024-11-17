/**
 * Studio module.
 */


import { Button, Box, Stack, Paper, Typography, Grid2 as Grid } from '@mui/material';
import { useState } from 'react';

import Viewport from '../../components/ui/viewport/Viewport';
import { URLForm, EEGNameForm } from '../../components/forms';
import { default as BlackSquare } from './BlackSquare';
import ConnectionSidebar from './ConnectionStatus';


// A single row of N eeg channel samples
type EEGDataChannel = [number, ...number[]];
// A set of NCHS eeg channels
type EEGDataPacket = EEGDataChannel[];

/**
 * @generic N The number of sample in the sent data frame.
 * @generic NCHS The number of channels of eeg data provided.
 * 
 * @param code The header code associated with a given stream message
 * @param data A multidimensional array of sample data, N per NCHS.
 * @param hz The samplerate of the underlying dataset.
 */
type EEGDataFrame = {
    code: "EMISSION" | "INIT" | "TERM" | "META";
    n: number;
    nchs: number;
    data: EEGDataPacket;
    hz: number;
}

/**
 * Defines a pooling of eeg data packets to collect client-side.
 */
class EEGBucket {

    // Default constructor
    constructor(pool?: EEGDataPacket[]) {
        if (pool === undefined) {
            this.pool = [];
        } else {
            this.pool = pool;
        }
    }

    // Sends the currently collected pool of data to the backend
    async send(name: string): Promise<number> {
        const eegData = {
            name: name,
            timeseries: this.pool.slice(0, 16),
            timestamps: this.pool[29],
        };

        const response = await fetch("/api/eeg/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(eegData)
        });

        if (response.status >= 400) {
            console.error(`Could not send pool data. Reason: ${response.status}`);
        } else {
            console.info("Successfully sent pool data");
        }

        return response.status;
    }

    pool: EEGDataPacket[];
}

/**
 * @param viewport Rendered viewport
 * @param websocket websocket connection endpoint
 */
type StudioProps = {
    websocket: WebSocket;
    bucket: EEGBucket;
    url: null | string;
    connectionStatus: "Disconnected" | "Connected" | "Connection Terminated",
    serialPort: string,
    deviceID: string,
    name: string,
}

/**
 * @return Studio Page.
 */
export default function Studio(): JSX.Element {
    const [studioState, setStudioState] = useState({
        websocket: new WebSocket("ws://localhost:8001"),
        bucket: new EEGBucket(),
        url: null,
        connectionStatus: "Disconnected",
        serialPort: "",
        deviceID: "",
        name: "",
    } as StudioProps);

    // onPlay handler for the viewport
    const startEEGStream = () => {
        const websocket = studioState.websocket;
        websocket.onopen = (_: Event) => {
            console.log("Socket Connected");
            websocket.send(JSON.stringify({code: "INIT", ivl: 1000}));
            setStudioState({
                ...studioState,
                connectionStatus: "Connected",
            })
        };

        websocket.onerror = (event: Event) => {
            console.error("Error: ", event);
            setStudioState({
                ...studioState,
                connectionStatus: "Connection Terminated",
            });
        };

        websocket.onmessage = (event: MessageEvent) => {
            const frame = JSON.parse(event.data);
            switch (frame.code) {
                case "META":
                    setStudioState({
                        ...studioState,
                        deviceID: frame.dongle_serial,
                        serialPort: frame.port,
                    });
                    break;
                case "EMISSION":
                    const mutatedBucket = studioState.bucket;
                    mutatedBucket.pool.push((frame as EEGDataFrame).data);
                    setStudioState({
                        ...studioState,
                        connectionStatus: "Connected",
                        bucket: studioState.bucket,
                    });
                    break;
                default:
                    break;
            }
        };

        websocket.onclose = (event: CloseEvent) => {
            console.log("Websocket connection closed: ", event);
            console.log("Code: ", event.code, "\nReason: ", event.reason);
            setStudioState({
                ...studioState,
                connectionStatus: "Disconnected",
            })
        };
    };

    // onPause handler for the viewport
    const stopEEGStream = () => {
        console.info("Webocket close event", studioState.websocket, studioState.bucket);
        if (studioState.websocket && studioState.bucket) {
            console.info("Closing websocket...");
            studioState.websocket.send(JSON.stringify({code: "TERM"}));
            studioState.bucket.send(studioState.name);
            studioState.websocket.close();
        }
    };

    // Save the url of the url form
    const saveUrl = (url?: string) => {
        if (url === undefined) {
            console.info("Invalid url submitted");
            return;
        }
        
        const youtubePattern = /https:\/\/(www.)?youtube.com/;
        const youtubeRE = new RegExp(youtubePattern);
        if (!youtubeRE.test(url)) {
            setStudioState({...studioState, url: ""});
        }

        const watchRE = /\/watch\?v=/;
        if (!watchRE.test(url)) {
            setStudioState({...studioState, url: ""});
        }

        console.info("Saving: ", url);
        setStudioState({...studioState, url: url});
    }

    const saveName = (name: string) => {
        console.info("Changing name: ", name)
        setStudioState({...studioState, name: name});
    }

    // Created at the start of the render, and never again.
    const startTime = new Date();

    return (
        <Box
            id="studio"
            sx={{
                padding: "10px",
                margin: "10px auto",
                height: "100%",
                width: "80%"
            }} component="div">
            <Grid
              container
              spacing={2}
              justifyContent="center"
              alignItems="center">
                <Grid size={8}>
                    <Paper
                        variant="outlined" elevation={2}
                        sx={{padding: "10px"}}>
                        <Box height="80vh">
                            <Stack flexDirection="row">
                                <URLForm label={"Video URL"} onChange={saveUrl}/>
                                <EEGNameForm onChange={saveName} />
                            </Stack>
                            { studioState.url &&
                                <Viewport
                                    url={studioState.url}
                                    onPlay={startEEGStream}
                                    onPause={stopEEGStream} />
                            }
                            { !studioState.url &&
                                <BlackSquare>
                                    <Typography
                                        variant="h6"
                                        sx={{color: "white", fontFamily: "monospace"}}>
                                        Please Insert Video URL
                                    </Typography>
                                </BlackSquare>
                            }
                        </Box>
                    </Paper>
                </Grid>
                <Grid size={4}>
                    <Paper
                        variant="outlined"
                        elevation={2}
                        sx={{padding: "10px"}}>
                        <ConnectionSidebar
                            connectionStatus={studioState.connectionStatus}
                            deviceId={studioState.deviceID}
                            serialPort={studioState.serialPort}
                            port={8001}
                            startTime={startTime}>
                            <Button id="start-session-button" onClick={startEEGStream}>
                                Start Session
                            </Button>
                        </ConnectionSidebar>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
}
