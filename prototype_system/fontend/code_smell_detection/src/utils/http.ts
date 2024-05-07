import getToken from './getToken';
import { message } from 'antd';
const initConfig = (config?: requestConfig) => {
  const headers = new Headers();
  //添加token
  getToken() ? headers.append('Authorization', getToken()) : null;
  if (config) {
    config.method = config.method || 'GET';
    config.headers = config.headers
      ? Object.assign(config.headers, headers)
      : headers;
    config.mode = config.mode || 'cors';
    config.credentials = config.credentials || 'include';
    if (config.method === 'POST' || config.method === 'PUT') {
      config.headers.append('Content-Type', 'application/json; charset=utf-8');
      if (config.body) config.body = JSON.stringify(config.body);
    }
    return config;
  } else {
    return {
      method: 'GET',
      headers: headers,
      mode: 'cors',
      credentials: 'include'
    };
  }
};
const checkStatus = (res: any) => {
  if (200 >= res.status && res.status < 300) {
    return res;
  }
  message.error(`服务器出错！${res.status}`);
  throw new Error(res.statusText);
};
const handleError = (error: any) => {
  if (error instanceof TypeError) {
    message.error(`网络请求失败啦！${error}`);
  }
  return {
    //防止页面崩溃，因为每个接口都有判断res.code以及data
    code: -1,
    data: false
  };
};
const judgeOkState = async (res: any) => {
  const cloneRes = await res.clone().json();
  //TODO:可以在这里管控全局请求
  if (!!cloneRes.code && cloneRes.code !== 200) {
    if (cloneRes.code === 403) localStorage.removeItem('token');
    message.error(`${cloneRes.msg}`);
  }
  return cloneRes;
};
type requestConfig = {
  method?: 'POST' | 'GET' | 'DELETE' | 'PUT';
  body?: any;
  headers?: any;
  mode?: any;
  credentials?: any;
  timeout?: number
};
const http = (url: string, config?: requestConfig) => {
  //初始化配置项
  const hadInitConfig = initConfig(config);
  const requestConfig = new Request(url, {...hadInitConfig,timeout:Infinity} as any);
  return fetch(requestConfig)
    .then(checkStatus)
    .then(judgeOkState)
    .catch(handleError);
};

export default http;
