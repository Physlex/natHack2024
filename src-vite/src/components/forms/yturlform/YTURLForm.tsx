/**
 * Youtube URL Form module.
 */


import { Box, TextField } from '@mui/material';


/**
 * @param onSubmit Event listener to allow parent components to use the url on submission.
//  */
// type YTURLFormParams = {
//     onSubmit: (url?: string) => void;
// }

/**
 * @returns The youtube url form.
 */
export default function YTURLForm(): JSX.Element {
    return (
        <Box component="form">
            <TextField id="outlined-basic" label="Outlined" variant="outlined" />
        </Box>
    );
}
