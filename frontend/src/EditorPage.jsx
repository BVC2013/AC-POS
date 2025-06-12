import React, { useState, useRef, useEffect } from 'react'
import Editor from '@monaco-editor/react'
import { API_URL } from './api'

// Update autocomplete and inline completion to use /autocomplete and correct payload
function EditorPage({ user, projectName, onBack }) {
  const [code, setCode] = useState('')
  const [loading, setLoading] = useState(false)
  const [output, setOutput] = useState('')
  const [pyodideReady, setPyodideReady] = useState(false)
  const pyodideRef = useRef(null)
  const editorRef = useRef(null)
  const monacoRef = useRef(null)
  const lastPromptRef = useRef('')
  const lastSuggestionRef = useRef('')

  // Load code from backend
  useEffect(() => {
    fetch(`${API_URL}/api/project?username=${user}&name=${projectName}`)
      .then(res => res.json())
      .then(proj => setCode(proj?.code || '# Start typing Python code...\n'))
  }, [user, projectName])

  // Save code to backend on change
  useEffect(() => {
    if (!code) return
    fetch(`${API_URL}/api/project/save`, {
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

  // Manual autocomplete button (do a lot more)
  const autocomplete = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/autocomplete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, max_tokens: 150 }), // much larger completion
      })
      const data = await res.json()
      // Optionally trim leading/trailing whitespace
      setCode(code + (data.completion ? data.completion.trim() : ''))
    } catch (err) {
      console.error('Autocomplete error:', err)
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
try:
${code.split('\n').map(line => '    ' + line).join('\n')}
    _result = sys.stdout.getvalue()
finally:
    sys.stdout = _stdout
_result
      `)
      setOutput(result)
    } catch (err) {
      setOutput(String(err))
    }
  }

  // Register inline suggestion provider (ghost autocomplete)
  const handleEditorWillMount = monaco => {
    monacoRef.current = monaco
    monaco.languages.registerInlineCompletionsProvider('python', {
      async provideInlineCompletions(model, position) {
        const textUntilPosition = model.getValueInRange({
          startLineNumber: 1,
          startColumn: 1,
          endLineNumber: position.lineNumber,
          endColumn: position.column,
        })

        if (lastPromptRef.current === textUntilPosition) {
          return { items: lastSuggestionRef.current ? [{
            text: lastSuggestionRef.current.trim(), // trim whitespace for ghost
            range: {
              startLineNumber: position.lineNumber,
              startColumn: position.column,
              endLineNumber: position.lineNumber,
              endColumn: position.column,
            },
          }] : [] }
        }

        lastPromptRef.current = textUntilPosition

        try {
          const res = await fetch(`${API_URL}/autocomplete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code: textUntilPosition, max_tokens: 32 }), // a bit more for ghost
          })
          const data = await res.json()
          lastSuggestionRef.current = data.completion ? data.completion.trim() : ''
          if (!data.completion) return { items: [] }
          return {
            items: [
              {
                text: data.completion.trim(),
                range: {
                  startLineNumber: position.lineNumber,
                  startColumn: position.column,
                  endLineNumber: position.lineNumber,
                  endColumn: position.column,
                },
              },
            ],
          }
        } catch {
          lastSuggestionRef.current = ''
          return { items: [] }
        }
      },
      handleItemDidShow: () => {},
      freeInlineCompletions: () => {},
    })
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
            onChange={value => setCode(value)}
            theme="vs-dark"
            options={{
              fontSize: 14,
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              wordWrap: 'on',
              fontFamily: 'Fira Code, monospace',
              inlineSuggest: { enabled: true },
            }}
            beforeMount={handleEditorWillMount}
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