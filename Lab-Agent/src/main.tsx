import * as React from 'react';
import * as ReactDOM from 'react-dom/client';
import { createBrowserRouter, RouterProvider } from 'react-router';
import App from './App';
import Layout from './layouts/dashboard';
import DashboardPage from './pages';
import UploadPage from './pages/upload';
import SignInPage from './pages/signin';
import UsersCrudPage from './pages/users';
import ChatBot from './pages/chatbot';

const router = createBrowserRouter([
  {
    Component: App,
    children: [
      {
        path: '/',
        Component: Layout,
        children: [
          {
            path: '',
            Component: DashboardPage,
          },
          {
            path: 'chat',
            Component: ChatBot,
          },
          {
            path: 'upload',
            Component: UploadPage,
          },
          {
            path: 'users/:userId?/*',
            Component: UsersCrudPage,
          },
        ],
      },
      {
        path: '/sign-in',
        Component: SignInPage,
      },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
