// import ErrorPage from '@/view/errorPage/errorPage.tsx';
import FilePath from '@/view/filePath/filePath.tsx';
import Index from '@/view/index/index.tsx';
import Result from '@/view/result/result.tsx';
import ShowTheCode from '@/view/showTheCode/showTheCode.tsx';
import { Navigate } from 'react-router-dom';

export default [
  {
    path: '/',
    element: <Index></Index>
  },
  {
    path: '/result',
    element: <Result></Result>
  },
  {
    path: '/filePath/:codeSmell',
    element: <FilePath></FilePath>
  },
  {
    path: '/showTheCode',
    element: <ShowTheCode></ShowTheCode>
  },
  {
    path: '*',
    element: <Navigate to="/" />
  }
];
