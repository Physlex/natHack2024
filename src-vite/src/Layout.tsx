/**
 * This module defines the layout for the rest of the application.
 */


import {
    Route,
    Outlet,
    createBrowserRouter,
    createRoutesFromElements
} from 'react-router-dom';

import { Studio, Diffuser, Login, Dashboard } from './pages';
import { Header, Footer } from './components/ui';
import { CssBaseline } from '@mui/material';

import { dashboardLoader } from './services/loaders';


/**
 * Layout for the application.
 */
export default function Layout(): JSX.Element {
    return (
        <div id="app">
            <CssBaseline />
            <Header />
            <Outlet />
            <Footer />
        </div>
    );
}

// Browser router to be used in the application
export const layoutRouter = createBrowserRouter(createRoutesFromElements(
    <Route element={<Layout />} >
        <Route path="/" element={<Login />} />
        <Route path="/studio" element={<Studio />} />
        <Route path="/diffuser" element={<Diffuser />} />
        <Route path="/dashboard" element={<Dashboard />} loader={dashboardLoader} />
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
});
