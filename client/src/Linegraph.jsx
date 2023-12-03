import React, { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

const Linegraph = ({ data }) => {
    const chartRef = useRef(null);
    const chartInstance = useRef(null);

    useEffect(() => {
        if (data.length > 0 && chartRef.current) {
            // Check if there's an existing chart instance
            if (chartInstance.current) {
                // If a chart instance exists, destroy it before creating a new one
                chartInstance.current.destroy();
            }

            const myChartRef = chartRef.current.getContext("2d");

            // Create the line chart
            chartInstance.current = new Chart(myChartRef, {
                type: 'line',
                data: {
                    labels: data.map((_, index) => `Data Point ${index + 1}`),
                    datasets: [{
                        label: 'Total Calorie',
                        data: data,
                        // backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        // borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Total Calorie'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Dates'
                            }
                        }
                    }
                }
            });
        }
    }, [data]);

    return (
        <div className="lineGraphMainContainer">
          <div className="lineGraphContainer"> 
              <canvas id="myChart" ref={chartRef} />
          </div>
        </div>
    );
};

export default Linegraph;
