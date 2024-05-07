import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname,'./src') //设置路径别名，需要引用/src下面的文件时只需要在前面添加@即可
    },
    extensions: ['.js', '.ts', '.json','tsx','jsx'] // 导入时想要省略的扩展名列表
  },
  server:{
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/')
      },
    }
  }
})
