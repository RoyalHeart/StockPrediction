import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
      colors: {
        c_green: "#73ff73",
        c_red: "#ff0f0a",
        c_orange: "#fda500",
      },
      keyframes: {
        tickerh: {
          "0%": {
            transform: "translate3d(100%, 0, 0)",
          },
          "100%": {
            transform: "translate3d(-550%, 0, 0)",
          },
        },
        tickerv: {
          "0%": {
            bottom: "0",
          } /* FIRST ITEM */,
          "20%": {
            bottom: "30px",
          } /* SECOND ITEM */,
          "40%": {
            bottom: "60px",
          } /* THIRD ITEM */,
          "60%": {
            bottom: "90px",
          } /* FORTH ITEM */,
          "80%": {
            bottom: "120px",
          } /* FORTH ITEM */,
          "100%": {
            bottom: "0;",
          } /* BACK TO FIRST */,
        },
      },
      animation: {
        tickerh: "tickerh linear 50s infinite",
        tickerv: "tickerv cubic-bezier(1, 0, 0.5, 0) 10s infinite",
      },
    },
  },
  plugins: [],
};
export default config;
