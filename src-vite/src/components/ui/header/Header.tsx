/**
 * Header Module.
 */


import { default as Navbar, NavbarButton } from './Navbar';


/**
 * @return Header component.
 */
export default function Header(): JSX.Element {
    return (
        <div id="header">
            <Navbar title="Dream Diffusion">
                <NavbarButton label="Studio" to="/studio" />
                <NavbarButton label="Dashboard" to="/dashboard" />
                <NavbarButton label="Diffuser" to="/diffuser" />
            </Navbar>
        </div>
    );
}
