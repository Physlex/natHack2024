/**
 * Studio module.
 */


import { Box } from '@mui/material';
import Viewport from '../../components/ui/viewport/Viewport';
import { URLForm } from '../../components/forms';
import { useState } from 'react';


/**
 * @param viewportUrl Youtube url assigned to the viewport on render.
 */
type StudioProps = {
    viewport: null | JSX.Element;
}

const startEEGStream = async () => {
    // TODO: The websocket here
}

/**
 * @return Studio Page.
 */
export default function Studio(): JSX.Element {
    const [studioState, setStudioState] = useState({
        viewport: null
    } as StudioProps);

    // Save the url of the url form
    const saveUrl = async (url?: string) => {
        if (url === undefined || url === null) {
            console.info("Invalid url submitted");
            return;
        }
        setStudioState({
            ...studioState,
            viewport: <Viewport url={url} onPlay={startEEGStream} />
        });
    }

    return (
        <Box id="studio" component="div">
            <URLForm label={"Video URL"} onSubmit={saveUrl}/>
            {studioState.viewport}
        </Box>
    );
}
