import { configureStore } from '@reduxjs/toolkit';
import myReducer from './myReducer/myReducer.tsx';


export default configureStore({
  reducer: {
    myReducer: myReducer
  },
  middleware: (getDefaultMiddleware) => getDefaultMiddleware()
});
