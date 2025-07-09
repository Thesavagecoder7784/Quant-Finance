
// Standard Normal cumulative distribution function
function normCdf(x) {
    return (1.0 + erf(x / Math.sqrt(2.0))) / 2.0;
}

// Error function
function erf(x) {
    // save the sign of x
    var sign = (x >= 0) ? 1 : -1;
    x = Math.abs(x);

    // A&S formula 7.1.26
    var a1 =  0.254829592;
    var a2 = -0.284496736;
    var a3 =  1.421413741;
    var a4 = -1.453152027;
    var a5 =  1.061405429;
    var p  =  0.3275911;

    // A&S formula 7.1.26
    var t = 1.0/(1.0 + p*x);
    var y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*Math.exp(-x*x);
    return sign*y; // erf(-x) = -erf(x)
}

// Standard Normal probability density function
function normPdf(x) {
    return (1.0 / (Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * x * x);
}

export function blackScholes(S, K, T, r, sigma, optionType = 'call') {
    if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0) {
        return { price: 0, delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
    }

    const d1 = (Math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);

    let price;
    let delta;
    let theta;
    let rho;

    if (optionType === 'call') {
        price = S * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2);
        delta = normCdf(d1);
        theta = (-S * normPdf(d1) * sigma / (2 * Math.sqrt(T)) - r * K * Math.exp(-r * T) * normCdf(d2));
        rho = K * T * Math.exp(-r * T) * normCdf(d2);
    } else if (optionType === 'put') {
        price = K * Math.exp(-r * T) * normCdf(-d2) - S * normCdf(-d1);
        delta = normCdf(d1) - 1;
        theta = (-S * normPdf(d1) * sigma / (2 * Math.sqrt(T)) + r * K * Math.exp(-r * T) * normCdf(-d2));
        rho = -K * T * Math.exp(-r * T) * normCdf(-d2);
    }

    const gamma = normPdf(d1) / (S * sigma * Math.sqrt(T));
    const vega = S * normPdf(d1) * Math.sqrt(T);

    return { price, delta, gamma, theta, vega, rho };
}
