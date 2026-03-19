'use client'
import { Component, ReactNode } from 'react'

export default class ErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean }
> {
  constructor(props: any) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-forge-black flex items-center justify-center flex-col gap-4 p-8 text-center">
          <div className="text-forge-accent text-4xl font-display">BEATHOLE AI</div>
          <p className="text-forge-muted">Something went wrong. Please refresh the page.</p>
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-2 bg-forge-accent text-white rounded-lg text-sm mt-2"
          >
            Refresh
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
