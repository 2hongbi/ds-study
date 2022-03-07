try:
    a = 10
    # b = 0
    b = 'zero'
    c = a / b
    print(c)
except ZeroDivisionError as e:
    print(e)
except TypeError as e:
    print(e)
except Exception as e:
    print(e)