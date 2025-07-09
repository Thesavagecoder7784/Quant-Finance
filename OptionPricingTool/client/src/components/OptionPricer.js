
import React from 'react';

const OptionPricer = ({ params, onParamChange }) => {
    return (
        <div className="option-pricer">
            <h2>Option Parameters</h2>
            <form>
                <div className="form-group">
                    <label>Stock Price (S)</label>
                    <input type="number" value={params.S} onChange={(e) => onParamChange('S', e.target.value)} />
                </div>
                <div className="form-group">
                    <label>Strike Price (K)</label>
                    <input type="number" value={params.K} onChange={(e) => onParamChange('K', e.target.value)} />
                </div>
                <div className="form-group">
                    <label>Time to Expiry (T)</label>
                    <input type="number" value={params.T} onChange={(e) => onParamChange('T', e.target.value)} />
                </div>
                <div className="form-group">
                    <label>Volatility (σ)</label>
                    <input type="number" value={params.sigma} onChange={(e) => onParamChange('sigma', e.target.value)} />
                </div>
                <div className="form-group">
                    <label>Risk-Free Rate (r)</label>
                    <input type="number" value={params.r} onChange={(e) => onParamChange('r', e.target.value)} />
                </div>
                <div className="form-group">
                    <label>Option Type</label>
                    <select value={params.optionType} onChange={(e) => onParamChange('optionType', e.target.value)}>
                        <option value="call">Call</option>
                        <option value="put">Put</option>
                    </select>
                </div>
            </form>
        </div>
    );
};

export default OptionPricer;
