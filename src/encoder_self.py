"""Byte pair encoding utilities"""

import os
import json
import regex as re
# 정규 표현식 (Regular Expression)
from functools import lru_cache
# functools : 고차 함수를 위한 것, 다른 함수에 작용하거나 다른 함수를 반환하는 함수

@lru_cache()
# cache() : 단순하고 가벼운 무제한 함수 캐시
# lru_cache() : (Least Recently Used) 함수의 리턴 값을 캐시, 최초 요청 이후에는 캐시된 결과를 리턴함

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.(예측할 수 없음)
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    # ord("!") 33 ~ ord("~") 126
    # [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 
    # ord("¡") 161 ~ ord("¬") 172
    # 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
    # ord("®") 174 ~ ord("ÿ") 255
    # 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
    # print(len(bs))  # 188개
    cs = bs[:]   # bs를 완전히 복사함
    # print(len(cs))  # 188개
    n = 0
    for b in range(2**8):   # 0 ~ 255 반복
        if b not in bs:     
            bs.append(b)    # 0 ~ 255 중에서 bs에 없는 숫자들을 bs에 넣는다.
            cs.append(2**8+n)   # 256 + n 을 cs에 넣는다.
            n += 1
    # print(n)        # 68개 증가함
    # print(len(cs))  # 256개
    # print(len(bs))  # 256개
    # [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 
    # 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 
    # 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 
    # 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323]
    cs = [chr(n) for n in cs]   # 숫자를 문자화한다.
    # print(cs)   # ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    return dict(zip(bs, cs))

# print(bytes_to_unicode())
# {33: '!', 34: '"', 35: '#', 36: '$', 37: '%', 38: '&', 39: "'", 40: '(', 41: ')', 42: '*', 43: '+', 44: ',', 45: '-', 46: '.', 47: '/', 48: '0', 49: '1', 50: '2', 51: '3', 52: '4', 53: '5', 54: '6', 55: '7', 56: '8', 57: '9'
# 해당 함수의 역할 : 숫자에 해당하는 문자를 dictionary로 묶어준다.


# byte_encoder = bytes_to_unicode()
# byte_decoder = {v:k for k, v in byte_encoder.items()}
# print(byte_decoder)
# {'!': 33, '"': 34, '#': 35, '$': 36, '%': 37, '&': 38, "'": 39, '(': 40, ')': 41, '*': 42, '+': 43, ',': 44, '-': 45, '.': 46, '/': 47, '0': 48, '1': 49, '2': 50, '3': 51, '4': 52, '5': 53, '6': 54, '7': 55, '8': 56, '9': 57,
# 원래 
# {33: '!', 34: '"', 35: '#', 36: '$', 37: '%', 38: '&', 39: "'", 40: '(', 41: ')', 42: '*', 43: '+', 44: ',', 45: '-',
# 디코더
# {'!': 33, '"': 34, '#': 35, '$': 36, '%': 37, '&': 38, "'": 39, '(': 40, ')': 41, '*': 42, '+': 43, ',': 44, '-': 45,


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).

    앞에 있는 글자와 뒤에 있는 글자들을 pair 시킴
    ('Ġ', 'w', 'a', 'n', 't') 를 넣었다면,
    G와 w / w와 a / a와 m / m와 t 끼리 결합시킴
    {('Ġ', 'w'), ('n', 't'), ('w', 'a'), ('a', 'n')}
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):  
        # encoder -> encoder.json 
        # bpe_merges -> vocab.bpe 튜플로 자른 거 
        # errors='replace' -> 인코딩의 규칙에 따라 입력 문자열을 변환할 수 없는 경우의 응답을 지정. 'replace' -> 마름모(◆)에 물음표(?)가 들어간 모습
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        # 원래 
        # {"!": 0, "\"": 1, "#": 2, "$": 3, "%": 4, "&": 5, "'": 6, "(": 7, ")": 8, "*": 9, "+": 10,
        # 디코더한 후, 
        # {0: '!', 1: '"', 2: '#', 3: '$', 4: '%', 5: '&', 6: "'", 7: '(', 8: ')', 9: '*', 10: '+'
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        # 숫자에 해당하는 문자를 dictionary로 묶어준다.
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        # 원래 
        # {33: '!', 34: '"', 35: '#', 36: '$', 37: '%', 38: '&', 39: "'", 40: '(', 41: ')', 42: '*', 43: '+', 44: ',', 45: '-',
        # 디코더한 후,
        # {'!': 33, '"': 34, '#': 35, '$': 36, '%': 37, '&': 38, "'": 39, '(': 40, ')': 41, '*': 42, '+': 43, ',': 44, '-': 45,
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # bpe_merges
        # [('Ġ', 't'), ('Ġ', 'a'), ('h', 'e'),
        # bpe_ranks
        # {('Ġ', 't'): 0, ('Ġ', 'a'): 1, ('h', 'e'): 2, 
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # 정규표현식
        # | 이거는 or 이라는 뜻 
        # ? 이거는 '모든 공간에서' 라는 뜻
        # + 이거는 '한 개 이상'
        # \p{L} : 어느 스크립트에 있는 단어 문자들
            # \p{L}+ : 모든 문자들
        # \p{N} : 어느 스크립트에 있는 숫자 문자들
            # \p{N}+ : 모든 숫자들
        # \s : 공백
        # \S : 공백이 아닌 모든 문자들
        # ^ : 새로운 줄
        # ?! : 앞의 패턴이 나타나지 않으면 이전의 패턴이 됩니다.
        # 예시)  “they’re” -> “they” and “‘re”
        # 그냥 문자 의미 단위별로 분리할 수 있는 거 같음


    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word) # 앞에 있는 글자와 뒤에 있는 글자들을 pair 시킴

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    # enc.encode(raw_text) 사용자가 입력한 text 문구가 들어간다.
    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):    # re.findall : 문자열 중 패턴과 일치되는 모든 부분을 찾는다.
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # print(token.encode('utf-8'))
            # b'I'
            # b' want'
            # b' to'
            # b' be'
            # b' a'
            # b' doctor'

            # print(token)
            # I
            # Ġwant
            # Ġto
            # Ġbe
            # Ġa
            # Ġdoctor 
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
byte_encoder = bytes_to_unicode()
with open(os.path.join('E:\\nmb\\gpt-2\\models\\1558M\\encoder.json'), 'r') as f:  # models_dir/model_name/encoder.json 경로에 있는 json 파일 오픈(r : 읽기모드) 
    encoder = json.load(f)  
with open(os.path.join('E:\\nmb\\gpt-2\\models\\1558M\\vocab.bpe'), 'r', encoding="utf-8") as f:   #bpe(Byte Pair Encoding) : 서브워드를 분리하는 알고리즘, 빈도수에 따라 문자를 병합하여 서브워드를 구성, 단어를 문자(char) 단위로 쪼갠 뒤, 가장 빈도수가 높은 쌍을 하나로 통합하는 과정을 반복
    bpe_data = f.read()                                                                     # 파일의 내용 전체를 문자열로 반환
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]

bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
cache = {}

def bpe(token):
    if token in cache:
        print('cache')
        return cache[token]
    word = tuple(token)
    # print(word) # ('Ġ', 'w', 'a', 'n', 't')
    pairs = get_pairs(word)
    print(pairs)    # {('Ġ', 'w'), ('n', 't'), ('w', 'a'), ('a', 'n')}

    if not pairs:   # 글자가 하나일 때
        return token
                                            # bpe_ranks : {('Ġ', 't'): 0, ('Ġ', 'a'): 1, ('h', 'e'): 2, 
    while True:
        bigram = min(pairs, key = lambda pair: bpe_ranks.get(pair, float('inf')))   

        # bpe_ranks.get(pair, float('inf') : key(pair)에 대응되는 value 반환 -> ('Ġ', 't') 0으로 반환, ('Ġ', 'a') 1로 반환...
        # float('inf') : pair 가 bpe_ranks에 없다면 inf 반환

        # key = lambda pair: bpe_ranks.get(('Ġ', 't'), float('inf'))
        # print(key)
        # <function bpe.<locals>.<lambda> at 0x000001FA657CF158>
        # <function bpe.<locals>.<lambda> at 0x000001FA657CF1E0>
        # <function bpe.<locals>.<lambda> at 0x000001FA657CF158>
        # <function bpe.<locals>.<lambda> at 0x000001FA657CF1E0>

        # print(bigram)
        # ('Ġ', 'w')
        # ('a', 'n')
        # ('an', 't')
        # ('Ġw', 'ant')

        if bigram not in bpe_ranks:
            break

        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:    # bigram 중 첫번째 위치에 있는 것
                j = word.index(first, i)    # word에서 first의 위치 찾기
                new_word.extend(word[i:j])
                print(first ,'->' , j, new_word)
                # Ġ  -> 0 []
                # a  -> 1 ['Ġw']
                # an -> 1 ['Ġw']
                # Ġw -> 0 []
                i = j
                # 여기서부터 코드 분석해야 함~~~~~!
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    word = ' '.join(word)
    cache[token] = word
    return word

bpe('Ġwant')


def encode(text):
        bpe_tokens = []
        for token in re.findall(pat, text):    # re.findall : 문자열 중 패턴과 일치되는 모든 부분을 찾는다.
            token = ''.join(byte_encoder[b] for b in token.encode('utf-8'))
            # print(token.encode('utf-8'))
            # b'I'
            # b' want'
            # b' to'
            # b' be'
            # b' a'
            # b' doctor'
            print(token)
            # I
            # Ġwant
            # Ġto
            # Ġbe
            # Ġa
            # Ġdoctor     
            bpe_tokens.extend(encoder[bpe_token] for bpe_token in bpe(token).split(' '))
            print(bpe_tokens)   # [40, 765, 284, 307, 257, 6253]
        return bpe_tokens

encode('I want to be a doctor')


# interactive_conditional_samples.py 43줄
def get_encoder(model_name, models_dir):
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:  # models_dir/model_name/encoder.json 경로에 있는 json 파일 오픈(r : 읽기모드) 
        encoder = json.load(f)                                                  # JSON 문자열을 Python의 객체로 변환
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:   # bpe(Byte Pair Encoding) : 서브워드를 분리하는 알고리즘, 빈도수에 따라 문자를 병합하여 서브워드를 구성, 단어를 문자(char) 단위로 쪼갠 뒤, 가장 빈도수가 높은 쌍을 하나로 통합하는 과정을 반복
        bpe_data = f.read()                                                                     # 파일의 내용 전체를 문자열로 반환
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]         # 맨 처음(#version: 0.2)이랑 맨 마지막(빈줄) 무시
    return Encoder(                 # Ecoder class로 들어간다.
        encoder=encoder,            # encoder.json
        bpe_merges=bpe_merges,      # vocab.bpe 튜플로 자른 거
    )


'''
with open(os.path.join('E:\\nmb\\gpt-2\\models\\1558M\\vocab.bpe'), 'r', encoding="utf-8") as f:   #bpe(Byte Pair Encoding) : 서브워드를 분리하는 알고리즘, 빈도수에 따라 문자를 병합하여 서브워드를 구성, 단어를 문자(char) 단위로 쪼갠 뒤, 가장 빈도수가 높은 쌍을 하나로 통합하는 과정을 반복
    bpe_data = f.read()                                                                     # 파일의 내용 전체를 문자열로 반환
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]

print(bpe_data.split('\n')[1:-1])
# ['Ġ t', 'Ġ a', 'h e', 'i n', 'r e', 'o n', 'Ġt he', 'e r', 'Ġ s', 'a t', 'Ġ w', 'Ġ o', 'e n', 'Ġ c',

count = 0
for mergestr in bpe_data.split('\n')[1:-1] :
    print(mergestr)
    # Ġ t
    # Ġ a
    # h e
    # i n
    # r e
    # o n
    # Ġt he
    # e r
    # Ġ s
    # a t
    # Ġ w
    # Ġ o
    # e n
    # Ġ c
    # i t
    # i s
    # a n
    # o r
    # e s
    # Ġ b

    print(tuple(mergestr.split()))
    # ('Ġ', 't')
    # ('Ġ', 'a')
    # ('h', 'e')
    # ('i', 'n')
    # ('r', 'e')
    # ('o', 'n')
    # ('Ġt', 'he')
    # ('e', 'r')
    # ('Ġ', 's')
    # ('a', 't')
    # ('Ġ', 'w')
    # ('Ġ', 'o')
    # ('e', 'n')
    # ('Ġ', 'c')
    # ('i', 't')
    # ('i', 's')
    # ('a', 'n')
    # ('o', 'r')
    # ('e', 's')
    # ('Ġ', 'b')

print(bpe_merges)
# [('Ġ', 't'), ('Ġ', 'a'), ('h', 'e'), ('i', 'n'), ('r', 'e'), ('o', 'n'), ('Ġt', 'he'), ('e', 'r'), ('Ġ', 's'), ('a', 't'), ('Ġ', 'w'), ('Ġ', 'o'), ('e', 'n'), ('Ġ', 'c'), ('i', 't'), ('i', 's'), ('a', 'n'), ('o', 'r'), ('e', 's'), ('Ġ', 'b'), ('e', 'd'), ('Ġ', 'f'), ('in', 'g'), ('Ġ', 'p'), ('o', 'u'), ('Ġa', 'n'), ('a', 'l'), ('a', 'r'), ('Ġt', 'o'), ('Ġ', 'm'), ('Ġo', 'f'), ('Ġ', 'in'), ('Ġ', 'd'), ('Ġ', 'h'), ('Ġan', 'd'), ('i', 'c'), ('a', 's'), ('l', 'e'), ('Ġt', 'h'), ('i', 'on'), ('o', 'm'), ('l', 'l'), ('en', 't'), ('Ġ', 'n'), ('Ġ', 'l'), ('s', 't'), ('Ġ', 're'), ('v', 'e'), ('Ġ', 'e'), ('r', 'o'), ('l', 'y'), ('Ġb', 'e'), ('Ġ', 'g'), ('Ġ', 'T'), ('c', 't'), ('Ġ', 'S'), ('i', 'd'), ('o', 't'), ('Ġ', 'I'), ('u', 't'), ('e', 't'), ('Ġ', 'A'), ('Ġ', 'is'), ('Ġ', 'on'), ('i', 'm'), ('a', 'm'), ('o', 'w'), ('a', 'y'), ('a', 'd'), ('s', 'e'), ('Ġth', 'at'), ('Ġ', 'C'), ('i', 'g'), ('Ġf', 'or'), ('a', 'c'), ('Ġ', 'y'), ('v', 'er'), ('u', 'r'), ('Ġ', 'u'), ('l', 'd'), ('Ġs', 't'), ('Ġ', 'M'), ("'", 's'), ('Ġ', 'he'), ('Ġ', 'it'), ('at', 'ion'), ('it', 'h'), ('i', 'r'), ('c', 'e'), ('Ġy', 'ou'), ('i', 'l'), ('Ġ', 'B'), 
'''

'''
with open(os.path.join('E:\\nmb\\gpt-2\\models\\1558M\\encoder.json'), 'r') as f:  # models_dir/model_name/encoder.json 경로에 있는 json 파일 오픈(r : 읽기모드) 
    encoder = json.load(f)  
decoder = {v:k for k,v in encoder.items()}
print(decoder)
# 원래 
# {"!": 0, "\"": 1, "#": 2, "$": 3, "%": 4, "&": 5, "'": 6, "(": 7, ")": 8, "*": 9, "+": 10,
# 디코더
# {0: '!', 1: '"', 2: '#', 3: '$', 4: '%', 5: '&', 6: "'", 7: '(', 8: ')', 9: '*', 10: '+'
'''

'''
with open(os.path.join('E:\\nmb\\gpt-2\\models\\1558M\\vocab.bpe'), 'r', encoding="utf-8") as f:   #bpe(Byte Pair Encoding) : 서브워드를 분리하는 알고리즘, 빈도수에 따라 문자를 병합하여 서브워드를 구성, 단어를 문자(char) 단위로 쪼갠 뒤, 가장 빈도수가 높은 쌍을 하나로 통합하는 과정을 반복
    bpe_data = f.read()                                                                     # 파일의 내용 전체를 문자열로 반환
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

print(bpe_ranks)
# {('Ġ', 't'): 0, ('Ġ', 'a'): 1, ('h', 'e'): 2, ('i', 'n'): 3, ('r', 'e'): 4, ('o', 'n'): 5, ('Ġt', 'he'): 6, ('e', 'r'): 7, ('Ġ', 's'): 8, ('a', 't'): 9, ('Ġ', 'w'): 10, ('Ġ', 'o'): 11, ('e', 'n'): 12, ('Ġ', 'c'): 13, ('i', 't'): 14, ('i', 's'): 15, ('a', 'n'): 16, ('o', 'r'): 17, ('e', 's'): 18, ('Ġ', 'b'): 19, ('e', 'd'): 20, ('Ġ', 'f'): 21, ('in', 'g'): 22, ('Ġ', 'p'): 23, ('o', 'u'): 24, ('Ġa', 'n'): 25, ('a', 'l'): 26, ('a', 'r'): 27, ('Ġt', 'o'): 28, ('Ġ', 'm'): 29, ('Ġo', 'f'): 30, ('Ġ', 'in'): 31, ('Ġ', 'd'): 32, ('Ġ', 'h'): 33, ('Ġan', 'd'): 34, ('i', 'c'): 35, ('a', 's'): 36, ('l', 'e'): 37, ('Ġt', 'h'): 38, ('i', 'on'): 39, ('o', 'm'): 40, ('l', 'l'): 41, ('en', 't'): 42, ('Ġ', 'n'): 43, ('Ġ', 'l'): 44, ('s', 't'): 45, ('Ġ', 're'): 46, ('v', 'e'): 47, ('Ġ', 'e'): 48, ('r', 'o'): 49, ('l', 'y'): 50, ('Ġb', 'e'): 51, ('Ġ', 'g'): 52, ('Ġ', 'T'): 53, ('c', 't'): 54, ('Ġ', 'S'): 55, ('i', 'd'): 56, ('o', 't'): 57, ('Ġ', 'I'): 58, ('u', 't'): 59, ('e', 't'): 60, ('Ġ', 'A'): 61, ('Ġ', 'is'): 62, ('Ġ', 'on'): 63, ('i', 'm'): 64, ('a', 'm'): 65, ('o', 'w'): 66, ('a', 'y'): 67, ('a', 'd'): 68, ('s', 'e'): 69, ('Ġth', 'at'): 70, ('Ġ', 'C'): 71, ('i', 'g'): 72, ('Ġf', 'or'): 73, ('a', 'c'): 74, ('Ġ', 'y'): 75, ('v', 'er'): 76, ('u', 'r'): 77, ('Ġ', 'u'): 78, ('l', 'd'): 79, ('Ġs', 't'): 80, ('Ġ', 'M'): 81, ("'", 's'): 82, ('Ġ', 'he'): 83, ('Ġ', 'it'): 84, ('at', 'ion'): 85, ('it', 'h'): 86, ('i', 'r'): 87, ('c', 'e'): 88, ('Ġy', 'ou'): 89, ('i', 'l'): 90, ('Ġ', 'B'): 91

# bpe_merges
# [('Ġ', 't'), ('Ġ', 'a'), ('h', 'e'),
# bpe_ranks
# {('Ġ', 't'): 0, ('Ġ', 'a'): 1, ('h', 'e'): 2, 
'''