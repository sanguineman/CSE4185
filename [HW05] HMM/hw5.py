import math
from collections import defaultdict, Counter
from math import log
import numpy as np

EPSILON = 1e-5

def smoothed_prob(arr, alpha=1):
    '''
    list of probabilities smoothed by Laplace smoothing
    input: arr (list or numpy.ndarray of integers which are counts of any elements)
           alpha (Laplace smoothing parameter. No smoothing if zero)
    output: list of smoothed probabilities

    E.g., smoothed_prob( arr=[0, 1, 3, 1, 0], alpha=1 ) -> [0.1, 0.2, 0.4, 0.2, 0.1]
          smoothed_prob( arr=[1, 2, 3, 4],    alpha=0 ) -> [0.1, 0.2, 0.3, 0.4]
    '''
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    _sum = arr.sum()
    if _sum:
        return ((arr + alpha) / (_sum + arr.size * alpha)).tolist()
    else:
        return ((arr + 1) / arr.size).tolist()

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    N = len(train)
    freq = defaultdict(list) # freq = {word : [], ...}
    tags = [] # 태그만 모아둔 리스트, 나중에 unseen word에 대한 처리를 위해 
    words = defaultdict(str) # words = {word : tag}, freq 딕셔너리를 바탕으로 최대 빈도 수 태그를 해당 단어와 매치해서 저장.
    for i in range(N):
        for w, t in train[i]: # train[i] is tuple / (word, tag)
            freq[w].append(t) # 해당 단어에 리스트에 매치된 태그를 전부 append
            tags.append(t) # 태그만 모아둔 리스트에 태그 append
    
    tags_freq = Counter(tags) # train data에 등장한 tag 빈도 수 세기 
    Max_tag = max(tags_freq, key=tags_freq.get) # train data에 나타난 가장 많은 빈도 수를 가진 태그, unseen word에 매치할 태그

    for word, tags in freq.items(): # 각 단어에 대한 태그의 빈도 수 처리
        tags_cnt = Counter(tags) # 각 단어의 리스트에 매치된 태그 전부가 들어가 있으므로, Counter를 통해 세기
        words[word] = max(tags_cnt, key=tags_cnt.get) # 최대 빈도 수 가진 태그를 바탕으로 words 딕셔너리에 word : tag 로 추가

    output = []
    M = len(test)
    for i in range(M):
        lst = []
        for w in test[i]:
            tup = None
            if w not in words: # if the corresponding word is not in the words dictionary, it means it is an unseen word
                tup = (w, Max_tag) # match it with the Max_tag
            else:
                tup = (w, words[w]) # match it with the words[w]
            lst.append(tup) # add it to the lst
        output.append(lst) # append the list to output list

    return output

    # raise NotImplementedError("You need to write this part!")


def viterbi(train, test):

    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # 문장 안에서 나오는 단어의 위치가 수업 시간에 배운 time 과 비슷한 개념이라고 생각하자. == 앞에 나온 단어가 뒤에 나온 단어에 영향을 미친다.
    # 구해야 할 것은 inital, emission, transition probability 이다. 

    '''
    1. Count tag => emission 구할 때 denominator 역할, tag pairs == (T_k-1, T_k), (word, tag) pairs
    '''
    N = len(train)
    words = [] # 등장한 단어들을 저장
    tags = [] # 등장한 태그들을 저장
    wtot = [] # (word, tag) pair
    ttot = [] # (T_k-1, T_k) pair

    for i in range(N):
        for j, (w, t) in enumerate(train[i]):
            words.append(w) # 단어 추가
            tags.append(t) # 태그 추가
            wtot.append((w,t)) # append word to tag pair
            if j > 0:
                ttot.append((train[i][j-1][1], train[i][j][1])) # append tag to tag pair
    
    words = Counter(words) # 각 단어 별 빈도 수
    tags = Counter(tags) # 각 태그 별 빈도 수 

    '''
    inital probability 구현
    '''
    initial = []
    for i in range(N):
        initial.append(train[i][0][1]) # 각 문장의 첫 번째 단어가 매칭된 tag append, 전부 "START"가 될 것이다.
    initial = Counter(initial) # Counter 딕셔너리로 변환
    initialf = []
    for t in tags.keys():
        if t in initial:
            initialf.append(initial[t]) # initial에 있다면, 해당 frequency를 append
        else:
            initialf.append(0) # initial에 없다면, 0을 추가
    inpb = {} # 태그 별 initial probability를 저장
    initialp = smoothed_prob(initialf, 0.001) # Laplace Smooth

    for i, t in enumerate(tags.keys()):
        inpb[t] = initialp[i] # 해당 태그의 initial probability를 저장
    
    '''
    transition probability 구현
    '''
    ttot = Counter(ttot) # (T_k-1, T_k) 별 빈도 수
    trpb = {}
    for t in tags.keys():
        denom = 0
        for tt in tags.keys():
            if (t, tt) in ttot:
                denom += (ttot[(t,tt)]) # 분모에 t가 오는 경우 frequency 더 하기
        for tt in tags.keys():
            if (t, tt) in ttot:
                trpb[(t,tt)] = (ttot[(t,tt)] + 0.001) / (denom + 0.001 * len(tags)) # Laplace smoothing
            else:
                trpb[(t,tt)] = 0.001 / (len(ttot)+ len(tags)* 0.001)


    '''
    emission probability 구현 
    '''

    # emission probability of one time word

    once = set() # 한번만 등장한 단어 저장
    for w, f in words.items():
        if f == 1:
            once.add(w)

    wtoto = {} # (w,t) 의 빈도 수 저장, 이 때 w는 한번만 등장한 단어이다.
    empbo = {} # emission probability of one time word
    wtot = Counter(wtot)
    total = 0
    for (w,t), f in wtot.items():
        if w in once:
            wtoto[(w,t)] = f # 한번만 등장한 단어의 경우 wtoto에 빈도 수 저장
            total += f

    for t in tags: # 태그 별 one time word에 대한 확률 분포 구하기
        Sum = 0
        for (w,tt) in wtoto:
            if tt == t:
                Sum += wtoto[(w,tt)]
        empbo[t] = (Sum + 0.001) / (total + 0.001 * len(wtoto)) # Laplace smoothing

    empb = {} # one time word 확률 분포 적용 emission probability
    for (w,t) in wtot:
        empb[(w,t)] = (wtot[(w,t)] + 0.001 * empbo[t]) / (tags[t] + 0.001 * len(tags) * empbo[t]) # one time word scaling with laplace smoothing

    unseen_em = {}
    for t in tags:
        unseen_em[t] =(0.001 * empbo[t]) / (tags[t] + 0.001 * empbo[t] * len(tags)) # one time word scaling with laplace smoothing for unseen words
    
    '''
    viterbi algorithm
    '''
    M = len(test)
    ans = []
    for i in range(M):
        forward = [] # save the probability for [index][tag]
        backward = [] # save the actual tag that is predicted 
        output = []
        for _ in range(len(test[i])):
            forward.append({t:0 for t in tags.keys()})
            backward.append({t:0 for t in tags.keys()})
        # 인덱스 0 번째 태그에 대한 전처리
        for t in tags.keys(): 
            ip = inpb[t]
            if (test[i][0], t) in empb: 
                ep = empb[(test[i][0],t)] # word to tag 가 있다면, empb에서 확률 반환
            else:
                ep = unseen_em[t] # 없으면, unknown_em에서 확률 반환
            forward[0][t] = log(ip) + log(ep) # 첫 번째 태그이므로, transition 이 없다.
        for j in range(1, len(test[i])):  # "START" 이후 첫번째 단어부터 predict
            for cur in tags.keys(): # 모든 태그를 돌아본다. 랜덤 변수 T가 가질 수 있는 도메인은 모든 태그이므로
                maxp = -100000000 # most likely tag probability
                pret = None # most likely tag

                if (test[i][j], cur) in empb: # 해당 word가 이미 있으면
                    emiss_prob = empb[(test[i][j],cur)] # 미리 구해둔 (word, tag) 확률 반환
                else:
                    emiss_prob = unseen_em[cur] # unseen word에 대해 one time word scaling을 적용한 emission probability 반환

                for prev in tags.keys(): # transition probability를 위해서, 이전 태그 iterate
                    trans_prob = trpb[(prev, cur)]
                    if log(emiss_prob) + log(trans_prob) + forward[j-1][prev] > maxp: # 모든 확률에 대해 log를 취한 후 더 한 것이 maxp보다 크면
                        maxp = log(emiss_prob) + log(trans_prob) + forward[j-1][prev]
                        pret = prev # 해당 태그 저장
                forward[j][cur] = maxp # 해당 확률 저장, 다음 j에서 활용될 것임. Dynamic programming 처럼.
                backward[j][cur] = pret # 나중에 backtrace을 하기 위해 태그 저장
        
        idx = len(test[i]) - 1 # 뒤에서 부터 backtrace
        max_key = max(forward[idx], key=forward[idx].get) # 문장의 마지막 단어에 대한 태그 가져오기 (최댓값)
        for j in range(idx,0,-1):
            output.append((test[i][j],max_key)) # output 리스트에 (word, 예측 태그) append
            max_key = backward[j][max_key] # backward에 저장해놓았으므로, j일 때 max_key를 만든 이 전 tag 반환
        output.append((test[i][0],max(forward[0], key=forward[0].get))) # 문장의 맨 처음 단어에 대한 태그 가져오기 (최댓값)
        output.reverse() # 역순이었으므로 다시 뒤집어 주기
        ans.append(output) # ans에 output 리스트 추가
    return ans
                    




    


    

    # raise NotImplementedError("You need to write this part!")