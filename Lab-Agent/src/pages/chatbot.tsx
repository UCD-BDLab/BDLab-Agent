import React, { useState, useEffect, useRef } from 'react';
import Box from '@mui/material/Box';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';
import SendIcon from '@mui/icons-material/Send';
import { API_BASE_URL } from '../config.js';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
// import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// import Link from '@mui/material/Link';

export default function ChatBot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text) return;

    setMessages(prev => [...prev, { sender: 'user', text }]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/qb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text }),
      });

      const payload = await res.json();

      if (!res.ok) {
        const message = payload?.error || res.statusText || 'Request failed';
        throw new Error(message);
      }

      const answer = payload?.answer ?? '';

      setMessages(prev => [...prev, { sender: 'bot', text: answer }]);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMessages(prev => [...prev, { sender: 'bot', text: `Error: ${message}` }]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <Box sx={{
        height: 800,
        width: '100%',
        boxShadow: 1,
        bgcolor: 'background.paper',
        border: 1,
        borderColor: 'divider',
        borderRadius: 1,
        p: 2,
        display: 'flex',
        flexDirection: 'column'}}>
      <Typography variant="h6" gutterBottom>
        BDLab Atlas
      </Typography>
      <Box sx={{ flexGrow: 1, overflowY: 'auto', mb: 1 }}>
        <List>
          {messages.map((m, idx) => (
            <ListItem key={idx} sx={{ justifyContent: m.sender === 'user' ? 'flex-end' : 'flex-start' }}>
              <Box sx={{ bgcolor: m.sender === 'user' ? 'primary.light' : 'grey.200', p: 1, borderRadius: 1, maxWidth: '80%' }}>
                {m.sender === 'user' ? (
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {m.text}
                </Typography>
              ) : (
                <Box sx={{ '& p': { m: 0, mb: 1 }, '& ul, & ol': { m: 0, pl: 3, mb: 1 } }}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {m.text ?? ''}
                  </ReactMarkdown>
                </Box>
              )}
              </Box>
            </ListItem>
          ))}
          <div ref={bottomRef} />
        </List>
      </Box>
      <Box sx={{ display: 'flex', gap: 1, mt: 'auto' }}>
        <TextField
          fullWidth
          placeholder="Type your message"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !loading) sendMessage();
          }}
        />
        <Button
          variant="contained"
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          endIcon={
            loading
              ? <CircularProgress size={16} color="inherit" />
              : <SendIcon />
          }
        >
          Send
        </Button>

      </Box>
    </Box>
  );
}
