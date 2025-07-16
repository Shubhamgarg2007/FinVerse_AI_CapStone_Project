// src/App.jsx
import React, { useState } from 'react';
import LoginForm from './components/LoginForm';
import SignupForm from './components/SignupForm';
import Dashboard from './components/Dashboard';
import './assets/styles/App.css';

function App() {
  const [currentView, setCurrentView] = useState('login'); // 'login', 'signup', 'dashboard'

  const showLogin = () => setCurrentView('login');
  const showSignup = () => setCurrentView('signup');
  const showDashboard = () => setCurrentView('dashboard');

  const renderView = () => {
    switch (currentView) {
      case 'signup':
        return <SignupForm onShowLogin={showLogin} onSignupSuccess={showDashboard} />;
      case 'dashboard':
        return <Dashboard onLogout={showLogin} />;
      case 'login':
      default:
        return <LoginForm onShowSignup={showSignup} onLoginSuccess={showDashboard} />;
    }
  };

  return (
    <div className="app-container">
      {currentView !== 'dashboard' && <h1 className="site-title">FinVerse AI</h1>}
      {renderView()}
    </div>
  );
}

export default App;