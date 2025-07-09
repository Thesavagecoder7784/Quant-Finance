
import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const PayoffChart = ({ params }) => {
    const { S, K, optionType } = params;

    const generatePayoffData = () => {
        const data = [];
        const labels = [];
        const startPrice = S * 0.8;
        const endPrice = S * 1.2;
        const step = (endPrice - startPrice) / 20;

        for (let price = startPrice; price <= endPrice; price += step) {
            labels.push(price.toFixed(2));
            let payoff;
            if (optionType === 'call') {
                payoff = Math.max(0, price - K);
            } else {
                payoff = Math.max(0, K - price);
            }
            data.push(payoff);
        }
        return { labels, data };
    };

    const { labels, data } = generatePayoffData();

    const chartData = {
        labels,
        datasets: [
            {
                label: `${optionType.charAt(0).toUpperCase() + optionType.slice(1)} Option Payoff`,
                data,
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
            },
        ],
    };

    const options = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Option Payoff Diagram',
            },
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Stock Price at Expiry',
                },
            },
            y: {
                title: {
                    display: true,
                    text: 'Profit / Loss',
                },
            },
        },
    };

    return <Line options={options} data={chartData} />;
};

export default PayoffChart;
