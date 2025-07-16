// src/components/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import '../assets/styles/Dashboard.css';
import ChatInterface from './ChatInterface';

// A small helper component for individual data points
const InfoWidget = ({ label, value, unit = '' }) => (
  <div className="widget">
    <h3>{label}</h3>
    <p className="value">
      {unit}{value}
    </p>
  </div>
);

const Dashboard = ({ onLogout }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

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
        console.error('Failed to fetch user data:', err);
        setError('Failed to load user data. Your session may have expired.');
        if (err.response && err.response.status === 401) {
          onLogout();
        }
      } finally {
        setLoading(false);
      }
    };

    fetchUserData();
  }, [onLogout]);

  if (loading) {
    return <div className="dashboard-container"><p>Loading your dashboard...</p></div>;
  }

  if (error) {
    return <div className="dashboard-container"><p className="error-message">{error}</p></div>;
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>Welcome, <span className="site-title-small">{user ? user.fullName.split(' ')[0] : 'User'}</span></h1>
        <button className="btn-logout" onClick={onLogout}>Logout</button>
      </div>

      <h2>Your Financial Profile</h2>
      <div className="dashboard-grid profile-grid">
        {user && (
          <>
            {/* FIX: Mapped field names to your probable user model for consistency */}
            <InfoWidget label="Annual Income" value={user.income?.toLocaleString() ?? 'N/A'} unit="₹" />
            <InfoWidget label="Monthly Savings" value={user.monthlySavings?.toLocaleString() ?? 'N/A'} unit="₹" />
            <InfoWidget label="Retirement Age" value={user.retirementAge ?? 'N/A'} />
            <InfoWidget label="Risk Appetite" value={user.riskAppetite ?? 'N/A'} />
            <InfoWidget label="Primary Goal" value={user.primaryGoal ?? 'N/A'} />
            <InfoWidget label="Investment Experience" value={user.investmentExperience ?? 'N/A'} />
          </>
        )}
      </div>
      <h2 style={{marginTop: '2rem'}}>AI Insights & Portfolio</h2>
      <div className="dashboard-grid insights-grid">
        {/* Placeholder widgets */}
        <div className="widget">
          <h3>Portfolio Snapshot</h3>
          <p>Your current portfolio value is:</p>
          <p className="value">$12,345.67</p>
        </div>
        <div className="widget">
          <h3>AI Recommendation</h3>
          <p>Based on market trends, we recommend adjusting your allocation. Consider increasing your position in <strong>SPY</strong>.</p>
        </div>
         <div className="widget">
          <h3>Market Analysis</h3>
          <p>The DQN model is currently in a 'hold' state, monitoring volatility in the tech sector (QQQ).</p>
        </div>
      </div>
        <h2 style={{marginTop: '2rem'}}>AI Chat Interface</h2>
        {/* FIX: Pass the fetched user data as a prop to ChatInterface. */}
        {/* This gives the chat component the necessary context. */}
        <ChatInterface user={user} />
    </div>
  );
};

export default Dashboard;