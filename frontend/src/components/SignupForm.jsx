import React, { useState } from 'react';
import '../assets/styles/Auth.css';
import axios from 'axios';

const SignupForm = ({ onShowLogin, onSignupSuccess }) => {
  
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    password: '',
    confirmPassword: '', 
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('--- Signup handleSubmit: Fired ---');

    // Password confirmation
    if (formData.password !== formData.confirmPassword) {
      alert('Passwords do not match!');
      return;
    }

    // Prepare data to send (only basic user details)
    const dataToSend = {
      fullName: formData.fullName,
      email: formData.email,
      password: formData.password,
      
      
      income: 0, 
      riskAppetite: 'Medium',
      primaryGoal: 'Wealth Creation',
      retirementAge: 65,
      monthlySavings: 0,
      investmentExperience: 'Beginner',
    };

    try {
      console.log('--- Signup handleSubmit: Inside try block, about to call API ---');
      console.log('--- Signup handleSubmit: Sending this data:', dataToSend);

      // Send the basic user data
      const response = await axios.post('http://localhost:8000/api/signup', dataToSend);

      console.log('--- Signup handleSubmit: Signup API call successful ---', response.data);

      // Proceed to obtain token
      const loginData = new URLSearchParams();
      loginData.append('username', formData.email);
      loginData.append('password', formData.password);

      const tokenResponse = await axios.post('http://localhost:8000/api/token', loginData);

      console.log('--- Signup handleSubmit: Token API call successful ---', tokenResponse.data);

      localStorage.setItem('token', tokenResponse.data.access_token);
      onSignupSuccess();
    } catch (err) {
      console.error('--- Signup handleSubmit: CRASHED in catch block ---', err);
      if (err.response && err.response.status === 422) {
        alert('Validation Error: Please check the data you entered. Details in console.');
        console.error('FastAPI Validation Error:', err.response.data.detail);
      } else if (err.response && err.response.data && err.response.data.detail) {
        alert(err.response.data.detail);
      } else {
        alert('An unexpected error occurred during signup.');
      }
    }
  };

  return (
    <div className="auth-form-container">
      <h2>Create Your Account</h2>
      {/* Removed step indicator */}
      <form onSubmit={handleSubmit}>
        <>
          <h3>Personal Details</h3>
          <div className="form-group">
            <label htmlFor="fullName">Full Name</label>
            <input
              type="text"
              id="fullName"
              name="fullName"
              className="form-input"
              value={formData.fullName}
              onChange={handleChange}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              name="email"
              className="form-input"
              value={formData.email}
              onChange={handleChange}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              className="form-input"
              value={formData.password}
              onChange={handleChange}
              required
              minLength="8"
            />
          </div>
          <div className="form-group">
            <label htmlFor="confirmPassword">Confirm Password</label>
            <input
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              className="form-input"
              value={formData.confirmPassword}
              onChange={handleChange}
              required
              minLength="8"
            />
          </div>
          <div className="form-navigation">
            {/* Removed "Back" and "Next" buttons, only submit remains */}
            <button type="submit" className="btn-primary" style={{ width: '100%' }}>
              Create Account
            </button>
          </div>
        </>
      </form>
      <div className="auth-switch-text">
        Already have an account?{' '}
        <button onClick={onShowLogin}>Login</button>
      </div>
    </div>
  );
};

export default SignupForm;