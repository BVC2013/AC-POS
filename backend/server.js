require('dotenv').config()
const express = require('express')
const mongoose = require('mongoose')
const cors = require('cors')
const bodyParser = require('body-parser')
const bcrypt = require('bcrypt')

const SALT_ROUNDS = 10

const app = express()
app.use(cors({
  origin: 'https://ac-pos.vercel.app'
}))
app.use(bodyParser.json())

// Use your provided MongoDB Atlas connection string
mongoose.connect(
  process.env.MONGODB_URI ||
    'mongodb+srv://bshivakoti08:Yash2014!@cluster0.ykixuh4.mongodb.net/blackide?retryWrites=true&w=majority&appName=Cluster0',
  {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  }
)

const UserSchema = new mongoose.Schema({
  username: String,
  password: String, // Store hashed password!
})
const ProjectSchema = new mongoose.Schema({
  user: String,
  name: String,
  code: String,
})
const User = mongoose.model('User', UserSchema)
const Project = mongoose.model('Project', ProjectSchema)

const usernamePattern = /^[a-zA-Z0-9_-]+$/

// Login or register
app.post('/api/login', async (req, res) => {
  const { username, password } = req.body
  if (!username || !usernamePattern.test(username) || !password) {
    return res.status(400).json({ error: 'Invalid username or password format' })
  }
  let user = await User.findOne({ username })
  if (!user) return res.status(400).json({ error: 'User not found' })
  if (!user.password) {
    return res.status(400).json({ error: 'This account was created before password support. Please register again with a new username.' })
  }
  const match = await bcrypt.compare(password, user.password)
  if (!match) return res.status(400).json({ error: 'Incorrect password' })
  res.json({ username: user.username })
})

// Register endpoint (fail if user exists)
app.post('/api/register', async (req, res) => {
  const { username, password } = req.body
  if (!username || !usernamePattern.test(username) || !password) {
    return res.status(400).json({ error: 'Invalid username or password format' })
  }
  let user = await User.findOne({ username })
  if (user) return res.status(400).json({ error: 'User already exists' })
  try {
    const hashed = await bcrypt.hash(password, SALT_ROUNDS)
    user = await User.create({ username, password: hashed })
    res.json({ username: user.username })
  } catch (err) {
    res.status(500).json({ error: 'Server error' })
  }
})

// List projects
app.get('/api/projects', async (req, res) => {
  const { username } = req.query
  const projects = await Project.find({ user: username })
  res.json(projects)
})

// Create project
app.post('/api/projects', async (req, res) => {
  const { username, name } = req.body
  const exists = await Project.findOne({ user: username, name })
  if (exists) return res.status(400).json({ error: 'Project exists' })
  const project = await Project.create({ user: username, name, code: '# New Python project\n' })
  res.json(project)
})

// Get project code
app.get('/api/project', async (req, res) => {
  const { username, name } = req.query
  const project = await Project.findOne({ user: username, name })
  res.json(project)
})

// Save project code
app.post('/api/project/save', async (req, res) => {
  const { username, name, code } = req.body
  await Project.updateOne({ user: username, name }, { code })
  res.json({ success: true })
})

const PORT = process.env.PORT || 8001
app.listen(PORT, () => console.log(`API running on http://localhost:${PORT}`))