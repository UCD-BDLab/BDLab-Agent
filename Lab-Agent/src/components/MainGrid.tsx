import React from 'react';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import QBChatBox from './CustomizedDataGrid';

export default function MainGrid() {
  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Grid>
        <Grid>
          <QBChatBox />
        </Grid>
      </Grid>
    </Box>
  );
}
