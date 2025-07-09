
import React from 'react';

const Results = ({ results }) => {
    if (!results) {
        return null;
    }

    return (
        <div className="results">
            <h2>Results</h2>
            <div className="result-item">
                <span>Option Price:</span>
                <span>{results.price.toFixed(4)}</span>
            </div>
            <div className="result-item">
                <span>Delta:</span>
                <span>{results.delta.toFixed(4)}</span>
            </div>
            <div className="result-item">
                <span>Gamma:</span>
                <span>{results.gamma.toFixed(4)}</span>
            </div>
            <div className="result-item">
                <span>Vega:</span>
                <span>{results.vega.toFixed(4)}</span>
            </div>
            <div className="result-item">
                <span>Theta:</span>
                <span>{results.theta.toFixed(4)}</span>
            </div>
            <div className="result-item">
                <span>Rho:</span>
                <span>{results.rho.toFixed(4)}</span>
            </div>
        </div>
    );
};

export default Results;
