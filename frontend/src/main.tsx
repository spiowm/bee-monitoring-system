import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { client } from './api/generated/client.gen'
import App from './App.tsx'
import './index.css'

client.setConfig({
  // VITE_API_URL set explicitly → use it (local dev or Kaggle two-tunnel)
  // VITE_API_URL="" (Docker / Lightning) → use current origin (same-origin serving)
  baseURL: import.meta.env.VITE_API_URL || window.location.origin,
});

const queryClient = new QueryClient();

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
)
