import * as React from 'react';
import {
  Box, Button, IconButton, LinearProgress, Stack, Typography, Tooltip,
} from '@mui/material';
import { DataGrid, GridColDef, GridRowModel, GridRowId } from '@mui/x-data-grid';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';
import {
  listFiles, uploadFile, deleteFile, renameFile,
} from '../firebase/storage';

type Row = {
  id: string;
  path: string;
  name: string;
  size: number;
  contentType: string;
  updated: string;
  url: string;
};

export default function UploadPage() {
  const [rows, setRows] = React.useState<Row[]>([]);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const folder = 'uploads';

  const load = React.useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const data = await listFiles(folder);
      setRows(data);
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load files');
    } finally {
      setBusy(false);
    }
  }, [folder]);

  React.useEffect(() => { load(); }, [load]);

  const onPick: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    setBusy(true);
    setError(null);
    try {
      for (const f of files) {
        await uploadFile(f, folder);
      }
      await load();
    } catch (e: any) {
      setError(e?.message ?? 'Upload failed');
    } finally {
      if (inputRef.current) inputRef.current.value = '';
      setBusy(false);
    }
  };

  const onDelete = async (id: GridRowId) => {
    const row = rows.find((r) => r.id === id);
    if (!row) return;
    if (!window.confirm(`Delete "${row.name}"?`)) return;
    setBusy(true);
    setError(null);
    try {
      await deleteFile(row.path);
      await load();
    } catch (e: any) {
      setError(e?.message ?? 'Delete failed');
    } finally {
      setBusy(false);
    }
  };

  const processRowUpdate = async (newRow: GridRowModel, oldRow: GridRowModel) => {
    const prev = oldRow as Row;
    const next = newRow as Row;
    if (prev.name !== next.name) {
      const res = await renameFile(prev.path, String(next.name).trim());
      return { ...next, name: res.name, path: res.path, id: res.path, url: res.url };
    }
    return next;
  };

  const columns = React.useMemo<GridColDef<Row>[]>(
    () => [
      { field: 'name', headerName: 'Name', flex: 1, editable: true },
      { field: 'contentType', headerName: 'Type', width: 160 },
      //simpler version
      // {
      //   field: 'size',
      //   headerName: 'Size',
      //   type: 'number',
      //   width: 120,
      //   valueFormatter: ({ value }) => `${(Number(value) / 1024).toFixed(1)} KB`,
      // },
      // {
      //   field: 'updated',
      //   headerName: 'Updated',
      //   width: 200,
      //   valueFormatter: ({ value }) => new Date(String(value)).toLocaleString(),
      // },
      //more robust version
      {
        field: 'size',
        headerName: 'Size',
        width: 120,
        valueFormatter: ({ value }) => {
          const bytes = Number(value) || 0;
          const kb = bytes / 1024;
          if (!Number.isFinite(kb)) return '0 KB';
          if (kb < 1024) return `${kb.toFixed(1)} KB`;
          const mb = kb / 1024;
          return `${mb.toFixed(1)} MB`;
        },
      },
      {
        field: 'updated',
        headerName: 'Updated',
        width: 200,
        valueFormatter: ({ value }) => {
          const d = new Date(String(value));
          return Number.isFinite(d.getTime()) ? d.toLocaleString() : '—';
        },
      },
      {
        field: 'actions',
        headerName: '',
        width: 120,
        sortable: false,
        filterable: false,
        renderCell: (params) => {
          const row = params.row as Row;
          return (
            <Stack direction="row" spacing={1}>
              <Tooltip title="Download">
                <IconButton
                  size="small"
                  component="a"
                  href={row.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <CloudDownloadIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Delete">
                <IconButton size="small" onClick={() => onDelete(row.id)}>
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Stack>
          );
        },
      },
    ],
    [rows],
  );

  return (
    <Box>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 1 }}>
        <Tooltip title="Refresh">
          <IconButton onClick={load} disabled={busy}><RefreshIcon /></IconButton>
        </Tooltip>
        <Box flex={1} />
        <input ref={inputRef} type="file" hidden multiple onChange={onPick} />
        <Button variant="contained" onClick={() => inputRef.current?.click()} disabled={busy}>
          + Add file
        </Button>
      </Stack>

      {busy && <LinearProgress sx={{ mb: 1 }} />}
      {error && <Typography color="error" sx={{ mb: 1 }}>{error}</Typography>}

      <div style={{ height: 520, width: '100%' }}>
        <DataGrid
          rows={rows}
          columns={columns}
          getRowId={(r) => r.id}
          disableRowSelectionOnClick
          processRowUpdate={processRowUpdate}
          onProcessRowUpdateError={(e) =>
            setError(e instanceof Error ? e.message : 'Rename failed')
          }
          pageSizeOptions={[5, 10, 25]}
          initialState={{
            pagination: { paginationModel: { page: 0, pageSize: 10 } },
            sorting: { sortModel: [{ field: 'updated', sort: 'desc' }] },
          }}
          experimentalFeatures={{ newEditingApi: true }}
        />
      </div>
    </Box>
  );
}
