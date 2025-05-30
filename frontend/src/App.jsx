import React, { useState } from 'react'
import Editor from '@monaco-editor/react'

function App() {
  const [code, setCode] = useState('# Start typing Python code...\n')
  const [loading, setLoading] = useState(false)

  const autocomplete = async () => {
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: code, max_new_tokens: 30 }),
      })
      const data = await res.json()
      setCode(code + data.completion)
    } catch (err) {
      console.error('Autocomplete error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-screen w-screen flex flex-col">
      <header className="p-4 bg-zinc-900 shadow-md text-xl font-semibold flex justify-between items-center">
        🧠 Black IDE
        <button
          onClick={autocomplete}
          disabled={loading}
          className="ml-4 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm disabled:opacity-50"
        >
          {loading ? 'Thinking...' : 'Autocomplete'}
        </button>
      </header>
      <main className="flex-1 bg-black p-2">
        <Editor
          height="100%"
          language="python"
          value={code}
          onChange={value => setCode(value)}
          theme="vs-dark"
          options={{
            fontSize: 14,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            wordWrap: 'on',
            fontFamily: 'Fira Code, monospace',
          }}
        />
      </main>
    </div>
  )
}

export default App
