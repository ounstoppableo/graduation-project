import { createSlice } from '@reduxjs/toolkit';

export const myReducerSlice = createSlice({
  name: 'resultStore',
  initialState: {
    data: {} as any
  },
  reducers: {
    setData: (state,action)=>{
      for(let key in action.payload.function){
        state.data[key] = action.payload.function[key]
      }
      for(let key in action.payload.class){
        state.data[key] = action.payload.class[key]
      }
    }
  }
});
// 每个 case reducer 函数会生成对应的 Action creators
export const { setData} = myReducerSlice.actions;

export default myReducerSlice.reducer;
