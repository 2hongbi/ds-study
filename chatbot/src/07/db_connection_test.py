# ex.7-1 (p.217) DB 호스트 연결 및 닫기
import pymysql

db = None
try:
    # DB 호스트 정보에 맞게 입력
    db = pymysql.connect(
        host='127.0.0.1',
        user='USER',
        password='PASSWORD',
        db='chatbot',
        charset='utf8'
    )
    print('DB 연결 성공')
except Exception as e:
    print(e)    # db 연결 실패 시 오류 내용 출력

finally:
    if db is not None:  # db가 연결된 경우에만 접속 닫기 시도
        db.close()
        print('DB 연결 닫기 성공')