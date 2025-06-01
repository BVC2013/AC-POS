import React, { useState } from 'react'
import LoginPage from './LoginPage'
import ProjectsPage from './ProjectsPage'
import EditorPage from './EditorPage'

function App() {
  const [user, setUser] = useState(null)
  const [project, setProject] = useState(null)

  const handleLogout = () => {
    setUser(null)
    setProject(null)
  }

  if (!user) {
    return <LoginPage onLogin={setUser} />
  }
  if (!project) {
    return (
      <ProjectsPage
        user={user}
        onOpenProject={setProject}
        onLogout={handleLogout}
      />
    )
  }
  return (
    <EditorPage
      user={user}
      projectName={project}
      onBack={() => setProject(null)}
    />
  )
}

export default App
