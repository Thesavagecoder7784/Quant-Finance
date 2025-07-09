import React, { useState, useEffect } from 'react';
import './App.css';
import OptionPricer from './components/OptionPricer';
import Results from './components/Results';
import PayoffChart from './components/PayoffChart';
import { blackScholes } from './utils/blackScholes';

function App() {
    const [params, setParams] = useState({
        S: 100,
        K: 100,
        T: 1,
        sigma: 0.2,
        r: 0.05,
        optionType: 'call',
    });

    const [results, setResults] = useState(null);

    const handleParamChange = (name, value) => {
        setParams(prevParams => ({
            ...prevParams,
            [name]: parseFloat(value) || value,
        }));
    };

    useEffect(() => {
        const { S, K, T, r, sigma, optionType } = params;
        if (S && K && T && r && sigma && optionType) {
            const calculatedResults = blackScholes(S, K, T, r, sigma, optionType);
            setResults(calculatedResults);
        }
    }, [params]);

    return (
        <div className="App">
            <header className="App-header">
                <h1>Option Pricing & Hedging Tool</h1>
            </header>
            <div className="container">
                <div className="pricer-container">
                    <OptionPricer params={params} onParamChange={handleParamChange} />
                </div>
                <div className="results-container">
                    <Results results={results} />
                </div>
                <div className="chart-container">
                    <PayoffChart params={params} />
                </div>
            </div>
        </div>
    );
}

export default App;