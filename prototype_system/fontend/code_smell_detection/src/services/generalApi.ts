import http from "@/utils/http.ts"

export const parserFile = (filePath:string) => {
    return http(`/api/parserFile`,{
        method:'POST',
        body: {
            filePath
        }
    })
}

export const detectSmell = (filePath:string)=>{
    return http(`/api/codeSmellDetect`,{
        method:'POST',
        body: {
            filePath
        }
    })
}

export const getCode = (filePath:string) =>{
    return http(`/api/getCode`,{
        method:'POST',
        body: {
            filePath
        }
    })
}