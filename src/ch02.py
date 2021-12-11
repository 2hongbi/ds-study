# 챗봇 클래스
class Chatbot:
    def say_hello(self):
        print('Say Hello')

    def say_name(self):
        print('My name is Kbot :D')


class SimpleObject:
    def __init__(self):
        print('call __init__()')

    def __del__(self):
        print('call __del__()')


if __name__ == '__main__':
    chatbot = Chatbot()
    chatbot.say_hello()
    chatbot.say_name()

    obj = SimpleObject()
    print('obj instance is alive ... ')
    del obj