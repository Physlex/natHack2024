/**
 * Footer module.
 */

/**
 * @return footer for the application.
 */
export default function Footer(): JSX.Element {
    return (
        <div id="footer">
            Copyright Dream Diffusion {new Date().getFullYear()}
        </div>
    );
}
