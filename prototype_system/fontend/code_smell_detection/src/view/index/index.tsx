import { detectSmell, parserFile } from "@/services/generalApi.ts"
import { useState } from "react"
import styles from './index.module.css';
import {Button, Input, notification} from 'antd'
import { setData } from "@/redux/myReducer/myReducer.tsx";
import { useDispatch } from "react-redux";
import { useNavigate } from 'react-router-dom';

function Index() {
    const [folderPath,setFolderPath] = useState('')
    const [parseButtonDisabled,setparseButtonDisabled] = useState(true)
    const [detectButtonDisabled,setDetectButtonDisabled] = useState(true)
    const [checkResultButtonDisabled,setCheckResultButtonDisabled] = useState(true)
    const [inputDisabled,setInputDisabled] = useState(false)
    const [parsing,setParsing] = useState(false)
    const [hadParse,setHadParse] = useState(false)
    const [hadDetect,setHadDetect] = useState(false)
    const [_folderPath,set_folderPath] = useState('')
    const dispatch = useDispatch();
    const navigate = useNavigate()
    const selectedFolder = (e:any)=>{
      if(e.currentTarget.value && !parsing) setparseButtonDisabled(false)
      else setparseButtonDisabled(true)
      setFolderPath(e.currentTarget.value)
    }
    const toParseFile = ()=>{
      setParsing(true)
      setFolderPath('')
      setparseButtonDisabled(true)
      parserFile(folderPath).then((res)=>{
        setParsing(false)
        if(res.code===200){
          setInputDisabled(true)
          setHadParse(true)
          setDetectButtonDisabled(false)
          set_folderPath(res.folderPath)
          notification.success({message:"解析文件成功！"})
        }else {
          setInputDisabled(false)
        }
      })
    }
    const toDetectSmell = ()=>{
      setDetectButtonDisabled(true)
      detectSmell(_folderPath).then(res=>{
        if(res.code===200){
          setHadDetect(true)
          setCheckResultButtonDisabled(false)
          notification.success({message:"检测坏味成功！"})
          dispatch(setData(res.result))
        }else {
          setInputDisabled(false)
          setDetectButtonDisabled(true)
          setHadParse(false)
        }
      })
    }

    const checkResult = ()=>{
      navigate('/result')
    }
  return (
    <>
      <div className={styles.card}>
        <h2>代码坏味检测工具</h2>
        <div className={styles.fileUploadBox}>
          <Input placeholder="请输入文件路径" disabled={inputDisabled} value={folderPath} onChange={(e)=>selectedFolder(e)}/>
          {hadParse?hadDetect?<Button type="primary" className={styles.success} disabled={checkResultButtonDisabled} onClick={checkResult}>查看结果</Button>:<Button type="primary" className={styles.detection} disabled={detectButtonDisabled} onClick={toDetectSmell}>检测坏味</Button>:<Button type="primary" disabled={parseButtonDisabled} onClick={toParseFile}>解析文件</Button>}
        </div>
        <div className={styles.warning}>
          <div>注意事项：</div>
          <div>1.必须上传根目录有__init__.py的文件夹。</div>
          <div>2.文件夹路径不能含有中文字符串。</div>
          <div>3.解析过程会比较久，请耐心等待。</div>
        </div>
      </div>
    </>
  )
}

export default Index
