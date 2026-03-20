/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx,js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          red: "#E3000F",
          "red-dark": "#B0000C",
          gray: "#3D3D3D",
        },
        bg: {
          base: "var(--bg-base)",
          card: "var(--bg-card)",
          elevated: "var(--bg-elevated)",
        },
        border: "var(--border-color)",
        status: {
          green: "#22C55E",
          amber: "#F59E0B",
          red: "#E3000F",
        },
        chart: {
          sim: "#E3000F",
          actual: "#3B82F6",
          grid: "#3F3F46",
        },
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        lg: "0.5rem",
        md: "calc(0.5rem - 2px)",
        sm: "calc(0.5rem - 4px)",
      },
    },
  },
  plugins: [],
};
