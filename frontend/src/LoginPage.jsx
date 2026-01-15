import React, { useState } from 'react'
import { BACKEND_API_URL } from './api'

function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [isRegister, setIsRegister] = useState(false)
  const [error, setError] = useState('')

  const usernamePattern = /^[a-zA-Z0-9_-]+$/

  const handleSubmit = async () => {
    if (!username.trim()) {
      setError('Username cannot be empty')
      return
    }
    if (!usernamePattern.test(username.trim())) {
      setError('Username can only contain letters, numbers, underscores, and dashes')
      return
    }
    if (!password) {
      setError('Password cannot be empty')
      return
    }
    try {
      const endpoint = isRegister ? '/api/register' : '/api/login'
      const res = await fetch(`${BACKEND_API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || (isRegister ? 'Registration failed' : 'Login failed'))
      onLogin(data.username)
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-zinc-900">
      <div className="w-full bg-amber-300 text-zinc-900 text-center text-sm font-semibold tracking-wide py-2">
        Scheduled downtime 12am to 8am - Autocomplete offline for model maintenance
      </div>
      <div className="flex-1 flex items-center justify-center">
        <div className="bg-zinc-800 p-8 rounded shadow-lg flex flex-col items-center">
          <h1 className="text-2xl font-bold mb-4 text-white">
            {isRegister ? 'Create Account' : 'Login to Black IDE'}
          </h1>
          <input
            className="mb-4 px-3 py-2 rounded bg-zinc-700 text-white"
            placeholder="Enter username"
            value={username}
            onChange={e => setUsername(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          />
          <input
            className="mb-4 px-3 py-2 rounded bg-zinc-700 text-white"
            placeholder="Enter password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          />
          <button
            className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-white font-semibold"
            onClick={handleSubmit}
          >
            {isRegister ? 'Create Account' : 'Login'}
          </button>
          <button
            className="mt-4 text-blue-400 underline"
            onClick={() => {
              setIsRegister(!isRegister)
              setError('')
            }}
          >
            {isRegister ? 'Already have an account? Login' : "Don't have an account? Create one"}
          </button>
          {error && <div className="text-red-400 mt-2">{error}</div>}
        </div>
      </div>
    </div>
  )
}

export default LoginPage