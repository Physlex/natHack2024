/**
 * Header Module.
 */


import { default as Navbar, NavbarButtonElement } from './Navbar';


/**
 * @return Header component.
 */
export default function Header(): JSX.Element {
    return (
        <div id="header">
            <Navbar>
                <NavbarButtonElement label="Studio" to="/studio" />
            </Navbar>
        </div>
    );
}
