# ex.7-5 (p.221) 데이터 삭제
import pymysql

db = None
try:
    db = pymysql.connect(
        host='127.0.0.1',
        user='USER',
        passwd='PASSWORD',
        db='chatbot',
        charset='utf8'
    )

    # 데이터 삽입 sql 정의
    id = 1  # 데이터 id(Primary key)
    sql = '''
        DELETE from tb_student where id=%d
    ''' % id

    # 데이터 삽입
    with db.cursor() as cursor:
        cursor.execute(sql)
    db.commit()

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()