/**
 * Module to define... a black square. Nice.
 */


import { Paper } from '@mui/material';
import { useEffect, useState } from 'react';


// Styles for the black square
const blackSquareStyles = {
    display: "flex",
    margin: "0 auto",
    flexGrow: 1,
    backgroundColor: "black",
    alignItems: "center",
    justifyContent: "center",
}

/**
 * Paramaters for the black square component. Yes, really.
 * @param { string | React.ReactNode } children See the react dom documentation on nested
 * react components.
 */
type BlackSquareParams = {
    children: string | React.ReactNode
}

/**
 * @returns Black Square. Nuff said.
 */
export default function BlackSquare({ children }: BlackSquareParams): JSX.Element {
    const [blackSquareState, setBlackSquareState] = useState({
        height: 360,
        width: 640
    });

    // Change the sizing to adapt in the *exact same way as the youtube video*
    const handleSizing = () => {
        const height = blackSquareState.height;
        const width = blackSquareState.width;

        const aspectRatio = 16/9;
        if (width / height > aspectRatio) {
            setBlackSquareState({
                ...blackSquareState,
                width: height * aspectRatio,
            });
        } else {
            setBlackSquareState({
                ...blackSquareState,
                height: width / aspectRatio,
            });
        }
    }

    useEffect(() => {
        handleSizing();

        window.addEventListener("resize", handleSizing);
        return (() => {window.removeEventListener("resize", handleSizing)});
    }, []);

    return (
        <Paper
            variant="outlined"
            sx={{
                ...blackSquareStyles,
                height: `${blackSquareState.height}px`,
                width: `${blackSquareState.width}px`,}}>
            {children}
        </Paper>
    );
}
