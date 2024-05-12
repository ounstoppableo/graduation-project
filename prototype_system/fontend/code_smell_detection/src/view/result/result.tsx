import { useSelector } from "react-redux";
import styles from './result.module.css';
import { useNavigate } from 'react-router-dom';
import { useEffect } from "react";
function Result(){
    const data = useSelector((state:any)=>state.myReducer.data)
    let items:any[] = []
    const navigate = useNavigate()
    const goToFilePathListPage = (codeSmell:string)=>{
        navigate(`/filePath/${codeSmell}`)
    }
    useEffect(()=>{
        if(JSON.stringify(data)==='{}') navigate('/') 
    },[])
    for(let key in data){
        let length = 0
        for(let i=0;i<Object.keys(data[key]).length;i++){
            length += data[key][Object.keys(data[key])[i]].length
        }
        items.push(<div onClick={()=>goToFilePathListPage(key)} style={{color:'#1890ff',cursor:'pointer'}} key={key}>{key}({length})</div>)
    }
    return <div className={styles.card}>
        {items}
    </div>
}
export default Result