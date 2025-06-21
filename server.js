const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

let waiting = null;

io.on('connection', socket => {
  console.log('User connected:', socket.id);

  if (waiting) {
    // Pair with waiting user
    const peer = waiting;
    waiting = null;
    socket.emit('matched', peer.id);
    peer.emit('matched', socket.id);
  } else {
    // Wait for a peer
    waiting = socket;
  }

  socket.on('signal', ({ to, data }) => {
    io.to(to).emit('signal', { from: socket.id, data });
  });

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
    if (waiting && waiting.id === socket.id) {
      waiting = null;
    }
    socket.broadcast.emit('peer-disconnected', socket.id);
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));