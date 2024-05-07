from flask import Flask
from flask import request
import json
from main import parsing_file,find_code_bad_smell
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

app = Flask(__name__)


@app.route('/parserFile', methods=['POST'])
def parser_file_api():
    if request.method == 'POST':
        filePath = request.json["filePath"]
        had_res = 0
        with ThreadPoolExecutor(max_workers=1) as executor:
           result_futures = executor.submit(parsing_file, filePath)
           try:
               result_futures.result()
           except Exception as e:
                if(str(e).startswith('错误的包文件夹！！')):
                    had_res=1
                    return {
                        "code": 201,
                        "msg": str(e)
                    }
                if(str(e).startswith('[WinError 183] 当文件已存在时，无法创建该文件。')):
                    had_res=1
                    return {
                        "code": 200,
                        "msg": 'success',
                        "folderPath":filePath,
                    }
                else:
                    had_res=1
                    return {
                        "code": 503,
                        "msg": '服务器错误！'
                    }
           finally:
               if(had_res==0):   
                    return {
                        "code": 200,
                        "msg": 'success',
                        "folderPath": filePath
                    }
        
@app.route('/codeSmellDetect', methods=['POST'])
def code_smell_detect_api():
    if request.method == 'POST':
        filePath = request.json["filePath"]
        try:
            res = find_code_bad_smell(filePath)
            return {
                "code": 200,
                "msg": "success",
                "result": res,
            }
        except Exception as e:
            print(e)
            return {
                "code": 503,
                "msg": "server error"
            }

@app.route('/getCode', methods=['POST'])
def get_code_api():
    if request.method == 'POST':
        filePath = request.json["filePath"]
        try:
            with open(filePath, 'r') as file:
                content = file.read()
                return {
                    "code": 200,
                    "msg": 'success',
                    "result": content
                }
        except Exception as e:
            print(e)
            return {
                "code": 503,
                "msg": "server error"
            }
