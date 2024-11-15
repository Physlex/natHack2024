/**
 * Studio module.
 */


import { Box } from '@mui/material';
import Viewport from '../../components/ui/viewport/Viewport';
import { YTURLForm } from '../../components/forms';
import { useState } from 'react';


/**
 * @param viewportUrl Youtube url assigned to the viewport on render.
 */
type StudioProps = {
    viewportUrl: string;
}

/**
 * @return Studio Page.
 */
export default function Studio(): JSX.Element {
    const [studioState, setStudioState] = useState({
        viewportUrl: ""
    } as StudioProps);

    // Save the url of the url form
    const saveUrl = async (url?: string) => {
        if (url === undefined) {
            throw new Error("Contact the developers, the youtube url form is broken");
        }
        setStudioState({...studioState, viewportUrl: url});
    }

    return (
        <Box id="studio" component="div">
            <YTURLForm label={"Video URL"} onSubmit={saveUrl}/>
            <Viewport url={studioState.viewportUrl} />
        </Box>
    );
}
