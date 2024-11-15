/**
 * Wrapper for youtube react component.
 */

import { Box } from '@mui/material';
import Youtube from 'react-youtube';
import { useState } from 'react';

type ViewportParams = {
    url: string;
}

/**
 * @prop The url to be played in the video.
 */
type ViewportProps = {
    url: string;
    opts: {
        height: string;
        width: string;
    };
}

/**
 * @return Viewport component.
 */
export default function Viewport({ url }: ViewportParams): JSX.Element {
    const [viewportState, setViewPortState] = useState({
        url,
        opts: {
            height: "640",
            width: "480",
        }
    } as ViewportProps);

    // On ready event handler for youtube api
    const onReady = async (event: any) => {
        event.target.pauseVideo();
    }

    try {
        new URL(url);
    } catch(error) {
        console.error(`Invalid url format for: ${url}`);
        setViewPortState({...viewportState, url: ""});
    }

    const youtubePattern = /https:\/\/(www.)?youtube.com/;
    const youtubeRE = new RegExp(youtubePattern);
    if (!youtubeRE.test(url)) {
        console.error(`Url ${url} is not a youtube url`);
        setViewPortState({...viewportState, url: ""});
    }

    const watchPattern = /\/watch\?v=/;
    const watchRE = new RegExp(watchPattern);
    let watchFirstIndex = -1;
    if (watchRE.test(url)) {
        watchFirstIndex = watchRE.lastIndex;
    } else {
        console.error(`Url has no watch parameter: ${url}`)
        setViewPortState({...viewportState, url: ""});
    }

    const videoID = url.slice(watchFirstIndex);
    console.info(`Video watch id: ${videoID}`);

    return (
        <Box className="viewport">
            <Youtube
                className="youtube-viewer"
                videoId={videoID}
                opts={viewportState.opts}
                onReady={onReady}
                >
            </Youtube>
        </Box>
    )
}
