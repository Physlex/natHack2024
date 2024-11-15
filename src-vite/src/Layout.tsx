/**
 * This module defines the layout for the rest of the application.
 */


import {
    Route,
    Outlet,
    createBrowserRouter,
    createRoutesFromElements
} from 'react-router-dom';

import { Studio, Diffuser } from './pages';


/**
 * Layout for the application.
 */
export default function Layout(): JSX.Element {
    return (
        <Header />
        <Outlet />
        <Footer />
    );
}

// Browser router to be used in the application
export const layoutRouter = createBrowserRouter(createRoutesFromElements(
    <Route element={<Layout />} >
        <Route path="/studio" element={<Studio />} />
        <Route path="/diffuser" element={<Diffuser />} />
    </Route>
),
{
    future: {
        v7_fetcherPersist: true,
        v7_normalizeFormMethod: true,
        v7_partialHydration: true,
        v7_relativeSplatPath: true,
        v7_skipActionErrorRevalidation: true,
    },
} 
);

