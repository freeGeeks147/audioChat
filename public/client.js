const socket = io();
let peerConnection;
const startBtn = document.getElementById('start');
const statusText = document.getElementById('status');

startBtn.onclick = async () => {
  startBtn.disabled = true;
  statusText.textContent = 'Finding a stranger...';
  socket.emit('find');
};

socket.on('matched', async peerId => {
  statusText.textContent = 'Matched! Connecting...';
  setupConnection(peerId);
});

socket.on('signal', async ({ from, data }) => {
  if (data.sdp) {
    await peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp));
    if (data.sdp.type === 'offer') {
      const answer = await peerConnection.createAnswer();
      await peerConnection.setLocalDescription(answer);
      socket.emit('signal', { to: from, data: { sdp: peerConnection.localDescription } });
    }
  } else if (data.candidate) {
    await peerConnection.addIceCandidate(data.candidate);
  }
});

socket.on('peer-disconnected', () => {
  statusText.textContent = 'Stranger disconnected.';
  cleanup();
});

async function setupConnection(peerId) {
  peerConnection = new RTCPeerConnection();
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  stream.getTracks().forEach(track => peerConnection.addTrack(track, stream));

  peerConnection.onicecandidate = e => {
    if (e.candidate) socket.emit('signal', { to: peerId, data: { candidate: e.candidate } });
  };

  peerConnection.onconnectionstatechange = () => {
    if (peerConnection.connectionState === 'connected') {
      statusText.textContent = 'You are now talking!';
    }
  };

  const offer = await peerConnection.createOffer();
  await peerConnection.setLocalDescription(offer);
  socket.emit('signal', { to: peerId, data: { sdp: peerConnection.localDescription } });
}

function cleanup() {
  if (peerConnection) {
    peerConnection.close();
    peerConnection = null;
  }
  startBtn.disabled = false;
}
