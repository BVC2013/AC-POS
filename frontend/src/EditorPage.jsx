import React, { useState, useRef, useEffect } from 'react'
import Editor from '@monaco-editor/react'
import { BACKEND_API_URL, AUTOCOMPLETE_API_URL } from './api'

function EditorPage({ user, projectName, onBack }) {
  const [code, setCode] = useState('')
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(false)
  const [pyodideReady, setPyodideReady] = useState(false)
  const pyodideRef = useRef(null)
  const editorRef = useRef(null)

  // Load code from backend
  useEffect(() => {
    fetch(`${BACKEND_API_URL}/api/project?username=${user}&name=${projectName}`)
      .then(res => res.json())
      .then(proj => setCode(proj?.code || '# Start typing Python code...\n'))
  }, [user, projectName])

  // Save code to backend on change
  useEffect(() => {
    if (!code) return
    fetch(`${BACKEND_API_URL}/api/project/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, name: projectName, code }),
    })
  }, [code, user, projectName])

  // Load Pyodide on mount
  useEffect(() => {
    if (!window.loadPyodide) return
    window.loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/" }).then(pyodide => {
      pyodideRef.current = pyodide
      setPyodideReady(true)
    })
  }, [])

  // Autocomplete button: send all code in the editor to backend
  const autocomplete = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${AUTOCOMPLETE_API_URL}/autocomplete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }), // sends all code in the editor
      })
      if (!res.ok) throw new Error('API error')
      const data = await res.json()
      setCode(code + (data.completion ? data.completion.trim() : ''))
    } catch (err) {
      alert('Autocomplete failed: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  // Run Python in browser using Pyodide
  const runInBrowser = async () => {
    if (!pyodideReady) {
      setOutput('Pyodide is still loading...')
      return
    }
    try {
      await pyodideRef.current.loadPackagesFromImports(code)
      let result = await pyodideRef.current.runPythonAsync(`
import sys
from io import StringIO
_stdout = sys.stdout
sys.stdout = StringIO()
${code}
_result = sys.stdout.getvalue()
sys.stdout = _stdout
_result
      `)
      setOutput(result)
    } catch (err) {
      setOutput(String(err))
    }
  }

  const handleEditorDidMount = (editor, monaco) => {
    editorRef.current = editor
  }

  return (
    <div className="h-screen w-screen flex flex-col">
      <header className="p-4 bg-zinc-900 shadow-md text-xl font-semibold flex justify-between items-center">
        <div>
          <button
            onClick={onBack}
            className="mr-4 px-3 py-1 bg-zinc-700 hover:bg-zinc-600 rounded text-sm text-white"
          >
            ← Projects
          </button>
          {projectName}
        </div>
        <div>
          <button
            onClick={autocomplete}
            disabled={loading}
            className="ml-4 px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm disabled:opacity-50"
          >
            {loading ? 'Thinking...' : 'Autocomplete'}
          </button>
          <button
            onClick={runInBrowser}
            disabled={!pyodideReady}
            className="ml-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm disabled:opacity-50"
          >
            Run in Browser
          </button>
        </div>
      </header>
      <main className="flex-1 bg-black p-2 flex flex-row">
        <div className="flex-1 pr-2">
          <Editor
            height="100%"
            language="python"
            value={code}
            onChange={value => value !== undefined && setCode(value)}
            theme="vs-dark"
            options={{
              fontSize: 14,
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              wordWrap: 'on',
              fontFamily: 'Fira Code, monospace',
            }}
            onMount={handleEditorDidMount}
          />
        </div>
        <div className="w-1/3 h-full flex flex-col">
          <div className="bg-zinc-900 text-green-400 font-mono rounded h-full p-3 overflow-auto shadow-inner border border-zinc-700">
            <div className="mb-2 text-white font-bold">Terminal</div>
            <pre className="whitespace-pre-wrap">{output}</pre>
          </div>
        </div>
      </main>
    </div>
  )
}

export default EditorPage