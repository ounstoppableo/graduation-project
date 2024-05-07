import { useSelector } from "react-redux";
import { useParams } from "react-router-dom";
import styles from './filePath.module.css';
import { useNavigate } from 'react-router-dom';
import { useEffect } from "react";

const FilePath = ()=>{
    const { codeSmell } = useParams();
    const data = useSelector((state:any)=>state.myReducer.data)
    const items = []
    const navigate = useNavigate()
    useEffect(()=>{
        if(JSON.stringify(data)==='{}') navigate('/') 
    },[])
    const showTheCode = (path:any)=>{
        navigate(`/showTheCode?path=${path}&smell=${codeSmell}`)
    }
    for(let i =0 ;i<Object.keys(data[codeSmell as any]).length;i++){
        if(data[codeSmell as any][Object.keys(data[codeSmell as any])[i]].length!==0) {
            items.push(<div onClick={()=>showTheCode(Object.keys(data[codeSmell as any])[i])} key={i}>{Object.keys(data[codeSmell as any])[i]}({data[codeSmell as any][Object.keys(data[codeSmell as any])[i]].length})</div>)
        }
    }
    return <div style={{color:'#1890ff',cursor:'pointer'}} className={styles.card}>{items}</div>
}
export default FilePath