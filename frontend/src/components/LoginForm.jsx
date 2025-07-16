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

    // For FastAPI's OAuth2, data must be sent as form-urlencoded.
    // URLSearchParams is the perfect tool for this.
    const params = new URLSearchParams();
    params.append('username', email); // The backend expects the email under the key 'username'.
    params.append('password', password);

    try {
      // console.log('--- Login handleSubmit: Inside try block, about to call API ---');
      // console.log('--- Login handleSubmit: Sending these params:', params.toString());

      // Make the POST request with the correct data and headers.
      // Axios automatically sets 'Content-Type': 'application/x-www-form-urlencoded' when you pass a URLSearchParams object.
      const res = await axios.post('http://localhost:8000/api/token', params);
      
      console.log('--- Login handleSubmit: API call successful ---', res.data);

      // Store the token and redirect to the dashboard.
      localStorage.setItem('token', res.data.access_token);
      onLoginSuccess();

    } catch (err) {
      console.error('--- Login handleSubmit: CRASHED in catch block ---', err);

      // Provide a user-friendly error message.
      if (err.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error('Error response data:', err.response.data);
        console.error('Error response status:', err.response.status);
        alert(err.response.data.detail || 'An error occurred during login.');
      } else if (err.request) {
        // The request was made but no response was received
        console.error('Error request:', err.request);
        alert('Could not connect to the server. Please check your network or try again later.');
      } else {
        // Something happened in setting up the request that triggered an Error
        console.error('Error message:', err.message);
        alert('An unexpected error occurred.');
      }
    }
  };

  return (
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
  );
};

export default LoginForm;