document.addEventListener("DOMContentLoaded", () => {
    // Simulate a sentiment score
    const sentimentScore = 65;

    // Display gauge
    const gauge = document.getElementById("gauge");
    gauge.innerHTML = `<div style="font-size: 2rem; color: ${getColor(sentimentScore)};">
        ${sentimentScore}
    </div>`;

    // Set up a basic chart using Chart.js
    const ctx = document.getElementById("sentiment-chart").getContext("2d");
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ["Jan", "Feb", "Mar", "Apr", "May"],
            datasets: [{
                label: 'Sentiment',
                data: [50, 55, 60, 65, sentimentScore],
                borderColor: 'blue',
                fill: false,
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { min: 0, max: 100 }
            }
        }
    });

    function getColor(score) {
        return score < 25 ? 'red' : score < 50 ? 'orange' : score < 75 ? 'yellow' : 'green';
    }
});
