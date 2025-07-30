import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import moment from 'moment';
import '../assets/styles/Dashboard.css';
import ChatInterface from './ChatInterface';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const InfoWidget = ({ label, value, unit = '' }) => (
  <div className="widget">
    <h3>{label}</h3>
    <p className="value">{unit}{value}</p>
  </div>
);

const Dashboard = ({ onLogout }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [chartLoading, setChartLoading] = useState(true);
  const [error, setError] = useState('');
  const [chartData, setChartData] = useState({ labels: [], datasets: [] });
  const [timeframe, setTimeframe] = useState('1year');

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          setError('Authentication token not found. Please log in again.');
          setLoading(false);
          return;
        }

        const response = await axios.get('http://localhost:8000/api/users/me', {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        setUser(response.data);
      } catch (err) {
        console.error('User data fetch failed:', err);
        setError('Failed to load user data. Your session may have expired.');
        if (err.response?.status === 401) {
          onLogout();
        }
      } finally {
        setLoading(false);
      }
    };

    fetchUserData();
  }, [onLogout]);

  useEffect(() => {
    const fetchMarketData = async () => {
      setChartLoading(true);
      try {
        const today = moment();
        let startDate;
        let dataIntervalDays;
        let dateFormat;
        const baseNSE = 23800;
        const baseBSE = 78500;
        const baseGold = 73000;

        switch (timeframe) {
          case '1week':
            startDate = moment().subtract(1, 'weeks');
            dataIntervalDays = 1;
            dateFormat = 'ddd, MMM D';
            break;
          case '1month':
            startDate = moment().subtract(1, 'months');
            dataIntervalDays = 1;
            dateFormat = 'MMM D';
            break;
          case '1year':
            startDate = moment().subtract(1, 'years');
            dataIntervalDays = 7;
            dateFormat = 'MMM YY';
            break;
          case '5years':
            startDate = moment().subtract(5, 'years');
            dataIntervalDays = 30;
            dateFormat = 'MMM YY';
            break;
          case 'all':
            startDate = moment().subtract(10, 'years');
            dataIntervalDays = 30;
            dateFormat = 'YYYY';
            break;
          default:
            startDate = moment().subtract(1, 'years');
            dataIntervalDays = 7;
            dateFormat = 'MMM YY';
        }

        const labels = [];
        const nseValues = [];
        const bseValues = [];
        const goldValues = [];

        let currentDate = moment(startDate);
        const totalDays = today.diff(startDate, 'days');
        const maxPoints = {
          '1week': 7,
          '1month': 30,
          '1year': 52,
          '5years': 60,
          'all': 100,
        }[timeframe] || 50;

        let i = 0;
        while (currentDate.isSameOrBefore(today) && i < maxPoints) {
          labels.push(currentDate.format(dateFormat));
          const progressFactor = currentDate.diff(startDate, 'days') / totalDays;
          const seasonality = Math.sin((2 * Math.PI * i) / (timeframe === '1year' ? 12 : 30));
          const fluctuationStrength = ['1week', '1month'].includes(timeframe) ? 0.008 : 0.004;

          const nse = baseNSE * (1 + 0.45 * progressFactor + 0.015 * seasonality + (Math.random() - 0.5) * fluctuationStrength);
          const bse = baseBSE * (1 + 0.4 * progressFactor + 0.012 * seasonality + (Math.random() - 0.5) * fluctuationStrength * 0.9);
          const gold = baseGold * (1 + 0.25 * progressFactor + 0.01 * seasonality + (Math.random() - 0.5) * fluctuationStrength * 0.7);

          nseValues.push(Math.round(nse));
          bseValues.push(Math.round(bse));
          goldValues.push(Math.round(gold));

          currentDate.add(dataIntervalDays, 'days');
          i++;
        }

        setChartData({
          labels,
          datasets: [
            {
              label: 'NSE Performance',
              data: nseValues,
              borderColor: 'rgb(75, 192, 192)',
              backgroundColor: 'rgba(75, 192, 192, 0.2)',
              tension: 0.3,
              fill: false,
              pointRadius: ['1week', '1month'].includes(timeframe) ? 3 : 0,
              pointHoverRadius: 5,
            },
            {
              label: 'BSE Performance',
              data: bseValues,
              borderColor: 'rgb(153, 102, 255)',
              backgroundColor: 'rgba(153, 102, 255, 0.2)',
              tension: 0.3,
              fill: false,
              pointRadius: ['1week', '1month'].includes(timeframe) ? 3 : 0,
              pointHoverRadius: 5,
            },
            {
              label: 'Gold Price (INR/10g)',
              data: goldValues,
              borderColor: 'rgb(255, 206, 86)',
              backgroundColor: 'rgba(255, 206, 86, 0.2)',
              tension: 0.3,
              fill: false,
              pointRadius: ['1week', '1month'].includes(timeframe) ? 3 : 0,
              pointHoverRadius: 5,
            },
          ],
        });
      } catch (err) {
        console.error('Market data fetch failed:', err);
        setError('Failed to load market data.');
      } finally {
        setChartLoading(false);
      }
    };

    fetchMarketData();
  }, [timeframe]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#ffffff',
          font: { size: 14 },
        },
      },
      title: {
        display: true,
        text: 'NSE, BSE, and Gold Performance',
        color: '#ffffff',
        font: { size: 20, weight: 'bold' },
      },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const value = ctx.parsed.y;
            return `${ctx.dataset.label}: ₹${value.toLocaleString()}`;
          },
        },
        backgroundColor: 'var(--surface-color)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: 'var(--border-color)',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        ticks: {
          color: '#ccc',
          maxRotation: 45,
          minRotation: 0,
        },
        grid: {
          color: '#444',
          borderColor: 'var(--border-color)',
        },
      },
      y: {
        ticks: {
          color: '#ccc',
          callback: (val) => '₹' + val.toLocaleString('en-IN', { maximumFractionDigits: 0 }),
        },
        grid: {
          color: '#444',
          borderColor: 'var(--border-color)',
        },
      },
    },
  }), [timeframe]);

  if (loading) return <div className="dashboard-container"><p>Loading your dashboard...</p></div>;
  if (error && !user) return <div className="dashboard-container"><p className="error-message">{error}</p></div>;

  return (
    <div className="dashboard-container">
      <div className="app-main-header">
        <img src="/logo.png" alt="FinVerse AI Logo" className="app-logo" />
        <div> 
          <h1 className="app-main-title">FinVerse AI</h1>
          <p className="app-main-tagline">Smart Investments, powered by AI</p>
        </div>
      </div>

      <div className="dashboard-header">
        <h1>Welcome, <span className="site-title-small">{user?.fullName?.split(' ')[0] || 'User'}</span></h1>
        <button className="btn-logout" onClick={onLogout}>Logout</button>
      </div>

      {/* AI Chat Interface */}
      <h2 style={{ marginTop: '2rem' }}>AI Chat Interface</h2>
      <ChatInterface user={user} />

      {/* Disclaimer Section */}
      <div className="disclaimer-box">
  <strong>Disclaimer:</strong> The AI predictions are based on historical data and may not reflect future performance. 
  Please consult a certified financial advisor before making any investment decisions.
</div>

      {/* Market Performance Graph Section */}
      <h2 style={{ marginTop: '2rem' }}>Market Performance Overview</h2>
      <div className="chart-controls">
        <label htmlFor="timeframe-select">View Performance Over:</label>
        <select
          id="timeframe-select"
          value={timeframe}
          onChange={(e) => setTimeframe(e.target.value)}
          className="timeframe-select"
        >
          <option value="1week">1 Week</option>
          <option value="1month">1 Month</option>
          <option value="1year">1 Year</option>
          <option value="5years">5 Years</option>
          <option value="all">All Years (Simulated)</option>
        </select>
      </div>

      <div className="chart-container-wrapper widget">
        {chartLoading ? (
          <p style={{ textAlign: 'center', color: 'var(--text-muted-color)' }}>Loading chart data...</p>
        ) : (
          <Line data={chartData} options={chartOptions} />
        )}
      </div>
      <p className="data-note">
 <strong> Note: </strong>The market performance data displayed above is simulated for illustrative and demonstration purposes and is not real-time.
</p>
    </div>
  );
};

export default Dashboard;
