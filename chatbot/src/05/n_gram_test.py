from konlpy.tag import Komoran


# 어절 단위 n-gram
def word_ngram(bow, num_gram):
    text = tuple(bow)
    ngrams = [text[x:x + num_gram] for x in range(0, len(text))]
    return tuple(ngrams)


# 유사도 계산
def similarity(doc1, doc2):
    cnt = 0
    for token in doc1:
        if token in doc2:
            cnt = cnt + 1
    return cnt/len(doc1)


if __name__ == '__main__':
    # 문장 정의
    sentence1 = '6월에 뉴턴은 선생님의 제안으로 트리니티에 입학했다.'
    sentence2 = '6월에 뉴턴은 선생님의 제안으로 대학교에 입학했디.'
    sentence3 = '나는 맛있는 밥을 뉴턴 선생님과 함께 먹었다.'

    # 형태소 분석기에서 명사(단어) 추출 > 명사를 리스트 형태로 추출
    komoran = Komoran()
    bow1 = komoran.nouns(sentence1)
    bow2 = komoran.nouns(sentence2)
    bow3 = komoran.nouns(sentence3)

    # 단어 n-gram 토큰 추출
    doc1 = word_ngram(bow1, 2)  # 2-gram 방식으로 추출
    doc2 = word_ngram(bow2, 2)
    doc3 = word_ngram(bow3, 2)

    # 추출된 n-gram 토큰 출력
    print(doc1)
    print(doc2)

    # 유사도 계산
    r1 = similarity(doc1, doc2)
    r2 = similarity(doc3, doc1)

    # 계산된 유사도 출력
    print(r1)
    print(r2)   # 문장 3과 문장 1은 연관성이 전혀 없음 > 유사도 0