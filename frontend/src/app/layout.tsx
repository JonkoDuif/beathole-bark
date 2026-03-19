import type { Metadata } from 'next'
import '@fontsource/bebas-neue'
import '@fontsource-variable/dm-sans'
import '@fontsource-variable/jetbrains-mono'
import { Toaster } from 'react-hot-toast'
import './globals.css'

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
}

export const metadata: Metadata = {
  title: 'Beathole AI — Generate & Sell Beats with AI',
  description: 'The #1 AI beat marketplace. Describe your sound, generate a professional instrumental in seconds, and start earning today.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <script dangerouslySetInnerHTML={{ __html: `
          var _chunkErr = false;
          var _chunkKey = 'chunkErrReload';
          function _handleChunkError() {
            _chunkErr = true;
            if (!sessionStorage.getItem(_chunkKey)) {
              sessionStorage.setItem(_chunkKey, '1');
              window.location.reload();
            }
          }
          window.addEventListener('error', function(e) {
            if (e.message && (e.message.indexOf('ChunkLoadError') !== -1 || e.message.indexOf('Loading chunk') !== -1)) {
              _handleChunkError();
            }
          });
          window.addEventListener('unhandledrejection', function(e) {
            if (e.reason && (e.reason.name === 'ChunkLoadError' || (e.reason.message && e.reason.message.indexOf('Loading chunk') !== -1))) {
              _handleChunkError();
            }
          });
          window.addEventListener('load', function() {
            if (!_chunkErr) {
              sessionStorage.removeItem(_chunkKey);
            }
          });
        `}} />
      </head>
      <body className="bg-forge-black text-forge-text font-body antialiased">
        {children}
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: '#111118',
              color: '#E8E8F0',
              border: '1px solid #1E1E2E',
              borderRadius: '12px',
              fontFamily: 'DM Sans, sans-serif',
            },
            success: {
              iconTheme: { primary: '#00FF88', secondary: '#111118' },
            },
            error: {
              iconTheme: { primary: '#FF3D6E', secondary: '#111118' },
            },
          }}
        />
      </body>
    </html>
  )
}
