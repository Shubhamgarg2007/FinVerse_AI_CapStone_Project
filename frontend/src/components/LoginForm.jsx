import React, { useState } from 'react';
import axios from 'axios';
import '../assets/styles/Auth.css'; 

const LoginForm = ({ onShowSignup, onLoginSuccess }) => {
  const [formData, setFormData] = useState({ email: '', password: '' });
  const { email, password } = formData;

  const onChange = (e) => setFormData({ ...formData, [e.target.id]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('--- Login handleSubmit: Fired ---');

    const params = new URLSearchParams();
    params.append('username', email);
    params.append('password', password);

    try {
      const res = await axios.post('http://localhost:8000/api/token', params);
      
      console.log('--- Login handleSubmit: API call successful ---', res.data);

      localStorage.setItem('token', res.data.access_token);
      onLoginSuccess();

    } catch (err) {
      console.error('--- Login handleSubmit: CRASHED in catch block ---', err);

      if (err.response) {
        console.error('Error response data:', err.response.data);
        console.error('Error response status:', err.response.status);
        alert(err.response.data.detail || 'An error occurred during login.');
      } else if (err.request) {
        console.error('Error request:', err.request);
        alert('Could not connect to the server. Please check your network or try again later.');
      } else {
        console.error('Error message:', err.message);
        alert('An unexpected error occurred.');
      }
    }
  };

  return (
    // The main container for the entire login page.
    
    <div className="login-page-wrapper"> 
      <div className="app-login-main-header">
        <img src="/logo.png" alt="FinVerse AI Logo" className="app-logo" />
        <div> 
          <h1 className="app-login-main-title">FinVerse AI</h1>
          <p className="app-login-main-tagline">Smart Investments, powered by AI</p>
        </div>
      </div>

      <div className="auth-form-container">
        <h2>Welcome Back</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input 
              type="email" 
              id="email" 
              className="form-input" 
              value={email} 
              onChange={onChange} 
              required 
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input 
              type="password" 
              id="password" 
              className="form-input" 
              value={password} 
              onChange={onChange} 
              required 
            />
          </div>
          <button type="submit" className="btn-primary">Login</button>
        </form>
        <div className="auth-switch-text">
          Don't have an account?{' '}
          <button onClick={onShowSignup}>Sign Up</button>
        </div>
      </div>
    </div>
  );
};

export default LoginForm;