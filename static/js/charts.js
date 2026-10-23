/**
 * Chart.js Visualizations Controller
 * Project: At-Risk Student Early Warning System Using Deep Neural Networks
 */

function initDashboardRiskDonut(canvasId, lowCount, medCount, highCount) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Low Risk", "Medium Risk", "High Risk"],
      datasets: [
        {
          data: [lowCount, medCount, highCount],
          backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
          borderColor: ["#ffffff", "#ffffff", "#ffffff"],
          borderWidth: 2,
          hoverOffset: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            font: { family: "-apple-system, sans-serif", size: 12 },
            padding: 16
          }
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const total = lowCount + medCount + highCount;
              const val = context.parsed;
              const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
              return ` ${context.label}: ${val} students (${pct}%)`;
            }
          }
        }
      },
      cutout: "68%"
    }
  });
}

function initDepartmentBarChart(canvasId, deptData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || !deptData) return;

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: deptData.labels,
      datasets: [
        {
          label: "Low Risk",
          data: deptData.low,
          backgroundColor: "#10b981",
          borderRadius: 4
        },
        {
          label: "Medium Risk",
          data: deptData.medium,
          backgroundColor: "#f59e0b",
          borderRadius: 4
        },
        {
          label: "High Risk",
          data: deptData.high,
          backgroundColor: "#ef4444",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          stacked: true,
          grid: { display: false }
        },
        y: {
          stacked: true,
          beginAtZero: true,
          ticks: { stepSize: 1 }
        }
      },
      plugins: {
        legend: {
          position: "bottom"
        }
      }
    }
  });
}

function initScatterPlot(canvasId, points) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || !points) return;

  const lowPoints = points.filter((p) => p.risk === "LOW RISK");
  const medPoints = points.filter((p) => p.risk === "MEDIUM RISK");
  const highPoints = points.filter((p) => p.risk === "HIGH RISK");

  new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Low Risk",
          data: lowPoints.map((p) => ({ x: p.x, y: p.y, name: p.name, id: p.id })),
          backgroundColor: "#10b981",
          pointRadius: 6,
          pointHoverRadius: 8
        },
        {
          label: "Medium Risk",
          data: medPoints.map((p) => ({ x: p.x, y: p.y, name: p.name, id: p.id })),
          backgroundColor: "#f59e0b",
          pointRadius: 6,
          pointHoverRadius: 8
        },
        {
          label: "High Risk",
          data: highPoints.map((p) => ({ x: p.x, y: p.y, name: p.name, id: p.id })),
          backgroundColor: "#ef4444",
          pointRadius: 7,
          pointHoverRadius: 9
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: "Attendance Percentage (%)", font: { weight: "bold" } },
          min: 30,
          max: 100
        },
        y: {
          title: { display: true, text: "Internal Assessment Marks (/100)", font: { weight: "bold" } },
          min: 15,
          max: 100
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: function (ctx) {
              const raw = ctx.raw;
              return `${raw.name} (${raw.id}): Att: ${raw.x}%, Marks: ${raw.y}`;
            }
          }
        }
      }
    }
  });
}
