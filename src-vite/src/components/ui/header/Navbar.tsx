/**
 * Navbar definition.
 */


import { AppBar, Box, Button, Toolbar, Typography } from '@mui/material';
import { Link } from 'react-router-dom';


const navBarLinkStyles = {
    display: { xs: 'none', md: 'flex' },
    fontFamily: 'monospace',
    fontWeight: 700,
    color: 'inherit',
    textDecoration: 'none',
};


/**
 * @param { string } label The label of the element
 * @param { string } to The link associated with the label
 * @param { boolean } hidden Whether or not this particular element is rendered
 */
type NavbarButtonElementParams = {
    label: string;
    to: string;
    isHidden?: boolean;
}

/**
 * @param { NavbarButtonElementParams } params See type definition for docs.
 * @returns A navbar label element.
 */
export function NavbarButtonElement({
    label,
    to,
    isHidden
}: NavbarButtonElementParams): JSX.Element {
    if (isHidden === undefined) {
        isHidden = false;
    }

    return (
        <>
            {!isHidden &&
            <Button color="inherit">
                <Link
                    to={to}>
                    <Typography
                        variant="h6"
                        noWrap
                        component="div"
                        align="center"
                        sx={navBarLinkStyles}>
                        {label}
                    </Typography>
                </Link>
            </Button>
            }
        </>
    );
}

type NavbarParams = {
    children: React.ReactNode;
}

/**
 * @returns Application navbar.
 */
export default function Navbar({ children }: NavbarParams): JSX.Element {
    return (
        <Box sx={{flexGrow: 1}}>
            <AppBar position="static">
                <Toolbar>
                    <Typography
                        variant="h6"
                        noWrap
                        sx={{flexGrow: 1}}
                        component="div">
                        Dream Diffusion
                    </Typography>
                    {children}
                </Toolbar>
            </AppBar>
        </Box>
    );
}
