/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#07080a',
        surface: {
          50: '#1a1d24',
          100: '#14161c',
          200: '#0e1015',
          300: '#090a0d',
        },
        border: {
          subtle: '#181b22',
          DEFAULT: '#222731',
          highlight: '#323947',
        },
        foreground: {
          DEFAULT: '#f3f4f6',
          muted: '#8b949e',
          subtle: '#525a66',
        },
        accent: {
          teal: '#00d2b4',
          cyan: '#38bdf8',
          gold: '#f59e0b',
          coral: '#f43f5e',
          amber: '#fbbf24',
        },
      },
      fontFamily: {
        serif: ['"Instrument Serif"', 'Georgia', 'serif'],
        sans: ['"IBM Plex Sans"', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'Menlo', 'monospace'],
      },
      letterSpacing: {
        widest: '.2em',
      },
      animation: {
        'pulse-subtle': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
