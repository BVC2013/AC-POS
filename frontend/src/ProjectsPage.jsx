import React, { useState, useEffect } from 'react'
import { API_URL } from './api'

function ProjectsPage({ user, onOpenProject, onLogout }) {
  const [projects, setProjects] = useState([])
  const [newProject, setNewProject] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${API_URL}/api/projects?username=${user}`)
      .then(res => res.json())
      .then(setProjects)
  }, [user])

  const handleCreate = async () => {
    if (!newProject.trim()) return
    try {
      const res = await fetch(`${API_URL}/api/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user, name: newProject.trim() }),
      })
      if (!res.ok) {
        setError('Project already exists')
        return
      }
      const proj = await res.json()
      setProjects([...projects, proj])
      setNewProject('')
      setError('')
    } catch {
      setError('Failed to create project')
    }
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-zinc-900">
      <header className="p-4 bg-zinc-800 flex justify-between items-center">
        <span className="text-xl text-white font-bold">Welcome, {user}</span>
        <button className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm text-white" onClick={onLogout}>
          Logout
        </button>
      </header>
      <main className="flex-1 flex flex-col items-center justify-center">
        <div className="bg-zinc-800 p-6 rounded shadow-lg w-full max-w-md">
          <h2 className="text-lg font-semibold mb-4 text-white">Your Projects</h2>
          <ul className="mb-4">
            {projects.map(proj => (
              <li key={proj.name} className="flex justify-between items-center mb-2">
                <span className="text-white">{proj.name}</span>
                <button
                  className="ml-2 px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs text-white"
                  onClick={() => onOpenProject(proj.name)}
                >
                  Open
                </button>
              </li>
            ))}
          </ul>
          <div className="flex">
            <input
              className="flex-1 px-2 py-1 rounded bg-zinc-700 text-white"
              placeholder="New project name"
              value={newProject}
              onChange={e => setNewProject(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleCreate()}
            />
            <button
              className="ml-2 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm text-white"
              onClick={handleCreate}
            >
              Create
            </button>
          </div>
          {error && <div className="text-red-400 mt-2">{error}</div>}
        </div>
      </main>
    </div>
  )
}

export default ProjectsPage