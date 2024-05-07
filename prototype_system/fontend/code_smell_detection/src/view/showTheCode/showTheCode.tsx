import { useLocation, useNavigate } from 'react-router-dom';
import styles from './showTheCode.module.css';
import { useEffect, useRef, useState } from 'react';
import { getCode } from '@/services/generalApi';
import 'highlight.js/styles/github.min.css';
import { useSelector } from 'react-redux';

const ShowTheCode = ()=>{
    const location = useLocation();
    const searchParams = new URLSearchParams(location.search);
    const data = useSelector((state:any)=>state.myReducer.data)
    const path = searchParams.get('path');
    const navigate = useNavigate()
    const dom = useRef<any>(null)
    const [content,setContent] = useState('')
    useEffect(()=>{
        if(JSON.stringify(data)==='{}') navigate('/') 
    },[])
    useEffect(()=>{
        getCode(path as string).then(res=>{
            if(res.code===200){
                //@ts-ignore
                setContent((window as any).hljs.highlight(res.result,
                    { language: 'python' }
                ).value)
                const smell = searchParams.get('smell');
                const locations = data[smell as any][path as any];
                setTimeout(()=>{
                   //@ts-ignore
                   (window as any).hljs.lineNumbersBlock(dom.current)
                   for(let i=0;i<dom.current.children[0].children[0].children.length;i++){
                       for(let j=0;j<locations.length;j++){
                           if(i+1>=locations[j]['start_line'] && i+1<=locations[j]['end_line']){
                               dom.current.children[0].children[0].children[i].style.backgroundColor = 'rgba(255,100,80,0.3)'
                           }
                       }
                   }
                },100)
            }
        })
    },[])
    return <div className={[styles.card,'python'].join(' ')}>
            <pre><code ref={dom} className="language-python" dangerouslySetInnerHTML={{ __html: content }}></code></pre>
        </div>
}
export default ShowTheCode