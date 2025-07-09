
# Option Pricing & Hedging Tool

This is an interactive web application that calculates the price of European call and put options using the Black-Scholes model. It also calculates and displays the common option Greeks.

## Features

- Prices European call and put options.
- Calculates the following Greeks: Delta, Gamma, Vega, Theta, and Rho.
- Visualizes the option's payoff diagram at expiration.
- Interactive UI to modify option parameters.

## Mathematical Model

The application uses the Black-Scholes model to price options. The model assumes that the market consists of at least one risky asset (e.g., a stock) and one riskless asset (e.g., a money market account).

The key parameters of the model are:

- **S**: The current price of the underlying asset.
- **K**: The strike price of the option.
- **T**: The time to expiration of the option (in years).
- **r**: The risk-free interest rate.
- **σ**: The volatility of the underlying asset's returns.

## How to Run Locally

1. Navigate to the `client` directory: `cd OptionPricingTool/client`
2. Install the dependencies: `npm install`
3. Start the development server: `npm start`
4. Open your browser and go to `http://localhost:3000`

## Deployment to GitHub Pages

To deploy this application to GitHub Pages, you can use the `gh-pages` package.

1. Install `gh-pages` as a dev dependency:
   ```bash
   npm install gh-pages --save-dev
   ```
2. Add the following properties to your `package.json` file:
   - `homepage`: `https://<your-github-username>.github.io/<your-repo-name>`
   - `predeploy`: `npm run build`
   - `deploy`: `gh-pages -d build`

   It should look like this:
   ```json
   {
     "name": "client",
     "version": "0.1.0",
     "private": true,
     "homepage": "https://Thesavagecoder7784.github.io/Quant-Finance",
     "dependencies": { ... },
     "scripts": {
       "start": "react-scripts start",
       "build": "react-scripts build",
       "test": "react-scripts test",
       "eject": "react-scripts eject",
       "predeploy": "npm run build",
       "deploy": "gh-pages -d build"
     },
     ...
   }
   ```
3. Deploy the application:
   ```bash
   npm run deploy
   ```
