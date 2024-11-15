/**
 * Entry point for vite to start putting things together.
 */


import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { default as Layout } from './Layout.tsx'

// Root of the application, what react uses to render
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Layout />
  </StrictMode>,
)
