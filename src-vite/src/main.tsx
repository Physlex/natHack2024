/**
 * Entry point for vite to start putting things together.
 */


import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router-dom';

import { layoutRouter } from './Layout.tsx'


// Root of the application, what react uses to render
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router = {layoutRouter} future={{v7_startTransition: true}}/>
  </StrictMode>,
)
