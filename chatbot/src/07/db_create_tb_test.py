# ex 7.2 (p.218) 테이블 생성
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

    # 테이블 생성 sql 정의
    sql = '''
    CREATE TABLE tb_student(
        id int primary key auto_increment not null,
        name varchar(32),
        age int,
        address varchar(32)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8
    '''

    # 테이블 생성
    with db.cursor() as cursor:
        cursor.execute(sql)     # execute : SQL 구문 실행
except Exception as e:
    print(e)
finally:
    if db is not None:
        db.close()