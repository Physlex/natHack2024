/**
 * Studio module.
 */


import { Box } from '@mui/material';
import Viewport from '../../components/ui/viewport/Viewport';
import { URLForm } from '../../components/forms';
import { useState } from 'react';


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
    code: "EMISSION" | "INIT" | "TERM";
    n: number;
    nchs: number;
    data: EEGDataPacket;
    hz: number;
}

/**
 * @param viewport Rendered viewport
 * @param websocket websocket connection endpoint
 */
type StudioProps = {
    websocket: null | WebSocket;
    url: null | string;
    bucket: null | EEGBucket;
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
    async send(): Promise<number> {
        const eegData = {
            timeseries: this.pool.slice(0, 16),
            timestamps: this.pool[30],
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
 * @return Studio Page.
 */
export default function Studio(): JSX.Element {
    const [studioState, setStudioState] = useState({
        websocket: null,
        url: null,
        bucket: null,
    } as StudioProps);

    // onPlay handler for the viewport
    const startEEGStream = () => {
        let websocket = new WebSocket("ws://localhost:8001");
        let bucket = new EEGBucket();

        setStudioState({
            ...studioState,
            bucket: bucket,
            websocket: websocket
        });

        websocket.onopen = (_: Event) => {
            console.log("Socket Connected");
            websocket.send(JSON.stringify({code: "INIT", ivl: 1000}));
        };

        websocket.onerror = (event: Event) => {
            console.error("Error: ", event);
        };

        websocket.onmessage = (event: MessageEvent) => {
            console.log("Message received: ", event.data);
            const frame = JSON.parse(event.data) as EEGDataFrame;
            if (frame.code === "EMISSION") {
                bucket.pool.push(frame.data);
                setStudioState({...studioState});
            }
        };

        websocket.onclose = (event: CloseEvent) => {
            console.log("Websocket connection close: ", event);
            console.log("Code: ", event.code, "\nReason: ", event.reason);
            bucket.send();
        };
    };

    // onPause handler for the viewport
    const stopEEGStream = () => {
        if (studioState.websocket !== null) {
            studioState.websocket.send(JSON.stringify({code: "TERM"}));
            studioState.websocket.close();
        }
    };

    // Save the url of the url form
    const saveUrl = async (url?: string) => {
        if (url === undefined) {
            console.info("Invalid url submitted");
            return;
        }
        setStudioState({...studioState, url: url});
    }

    return (
        <Box id="studio" sx={{
                paddingTop: "10px",
                margin: "0px auto",
                height: "70vh",
                width: "90%"
            }} component="div">
            <URLForm label={"Video URL"} onSubmit={saveUrl}/>
            { studioState.url &&
                <Viewport
                    url={studioState.url}
                    onPlay={startEEGStream}
                    onPause={stopEEGStream} />
            }
        </Box>
    );
}
