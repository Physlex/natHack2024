/**
 * Wrapper for youtube react component.
 */

import { Box } from '@mui/material';
import Youtube from 'react-youtube';
import { useEffect, useState } from 'react';

const youtubeViewerStyles = {
    display: "flex",
    margin: "0 auto",
    alignItems: "center",
    width: "100%",
    height: "100%",
    justifyContent: "center",
};

/**
 * @param url The video url to play
 * @param onPlay a hook to allow external components to drive the behaviour of the
 * viewport
 */
type ViewportParams = {
    url: string;
    onPlay: () => void;
}

/**
 * @prop url The url to be played in the video.
 * @prop videoID the watch id of the video url
 * @prop opts the height and width of the viewport
 */
type ViewportProps = {
    url: string;
    videoID: string;
    opts: {
        height: string;
        width: string;
    };
}

/**
 * @return Viewport component.
 */
export default function Viewport({ url, onPlay }: ViewportParams): JSX.Element {
    const [viewportState, setViewportState] = useState({
        url: url,
        videoID: "",
        opts: {
            height: "360",
            width: "640",
        }
    } as ViewportProps);

    // On ready event handler for youtube api
    const onReady = async (event: any) => {
        event.target.pauseVideo();
    }

    // Ensure the url is correct, setting the viewport state as necessary.
    const handleUrl = async (url:string) => {
        if (url === "") { // There isn't a youtube video to scan yet.
            console.info("Failed to find a video url");
            return;
        }

        try {
            new URL(url);
        } catch(error) {
            console.error(`Invalid url format for: ${url}`);
            setViewportState({...viewportState, url: ""});
        }

        const youtubePattern = /https:\/\/(www.)?youtube.com/;
        const youtubeRE = new RegExp(youtubePattern);
        if (!youtubeRE.test(url)) {
            console.error(`Url ${url} is not a youtube url`);
            setViewportState({...viewportState, url: "", videoID: ""});
        }

        const watchRE = /\/watch\?v=/;
        let watchIndex = -1;
        if (watchRE.test(url)) {
            const indexMatch = "/watch?v=";
            watchIndex = url.indexOf(indexMatch) + indexMatch.length;
        } else {
            console.error(`Url has no watch parameter: ${url}`)
            setViewportState({...viewportState, url: "", videoID: ""});
        }

        const videoID = url.slice(watchIndex);
        setViewportState({...viewportState, url: url, videoID: videoID});
    }

    // Ensure that the size is valid for the viewport.
    const handleSizing = async () => {
        const heightStr: string = viewportState.opts.height;
        const widthStr: string = viewportState.opts.width;
        
        const height: number = parseInt(heightStr);
        const width: number = parseInt(widthStr);
        const aspectRatio = 16/9;
        if (width / height > aspectRatio) {
            setViewportState({
                ...viewportState,
                opts: {
                    ...viewportState.opts,
                    width: String(height * aspectRatio)
                }
            })
        } else {
            setViewportState({
                ...viewportState,
                opts: {
                    ...viewportState.opts,
                    height: String(width / aspectRatio)
                }
            })
        }
    }

    useEffect(() => {
        handleSizing();
        handleUrl(url);

        window.addEventListener("resize", handleSizing);
        return (() => {window.removeEventListener("resize", handleSizing)});
    }, [url]);

    console.info("viewport video id: ", viewportState.videoID);

    return (
        <Box 
            className="viewport"
            sx={youtubeViewerStyles}>
            <Youtube
                className="youtube-viewer"
                videoId={`${viewportState.videoID}`}
                opts={viewportState.opts}
                onReady={onReady}
                onPlay={onPlay}
                >
            </Youtube>
        </Box>
    )
}
