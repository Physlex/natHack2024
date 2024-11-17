/**
 * Navbar definition.
 */


import { AppBar, Link, Box, Button, Toolbar, Typography } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';


// What is said on the tin
const navBarLinkStyles = {
    display: { xs: 'none', md: 'flex' },
    fontFamily: 'monospace',
    fontWeight: 700,
    color: 'white',
    textDecoration: 'none',
    textTransform: 'none',
};

/**
 * @param { string } label The label of the element
 * @param { string } to The link associated with the label
 * @param { boolean } hidden Whether or not this particular element is rendered
 */
type NavbarButtonParams = {
    label: string;
    to: string;
    isHidden?: boolean;
}

/**
 * @param { NavbarButtonParams } params See type definition for docs.
 * @returns A navbar label element.
 */
export function NavbarButton({
    label,
    to,
    isHidden
}: NavbarButtonParams): JSX.Element {
    if (isHidden === undefined) {
        isHidden = false;
    }

    return (
        <>
            {!isHidden &&
            <Button color="inherit">
                <Link component={RouterLink} to={to}>
                    <Typography
                        variant="h6"
                        noWrap
                        component="div"
                        align="center"
                        paddingRight="20px"
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
    title: string;
    children: React.ReactNode;
}

/**
 * @returns Application navbar.
 */
export default function Navbar({ title, children }: NavbarParams): JSX.Element {
    return (
        <Box sx={{flexGrow: 1}}>
            <AppBar position="static">
                <Toolbar>
                    <Typography
                        variant="h6"
                        noWrap
                        sx={{flexGrow: 1, ...navBarLinkStyles}}
                        component="div">
                        {title}
                    </Typography>
                    {children}
                </Toolbar>
            </AppBar>
        </Box>
    );
}
