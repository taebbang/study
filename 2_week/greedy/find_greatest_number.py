def solution(number, k):
    # number text의 길이를 받아준다
    length = len(number)
    i = 0
    # k가 0보다 크고 i가 length -1 갈때까지 while을 돌린다
    while i < length - 1 and k > 0:
        #문자를 처음부터 하나씩 체크하면서 다음 문자와 비교한다.
        #만약 첫문자가 다음 문자보다 작은경우 첫번째 문자를 제외시킨다
        #맨 앞에서부터 작은 숫자를 하나씩 제외시키면서 k개만큼 없애준다
        if number[i] < number[i+1]:
            if i != 0: 
                number = number[:i] + number[i+1:]
                length -= 1 # 길이감소
                k -= 1
                i -= 1
            else: 
                number = number[:i] + number[i + 1:]
                length -= 1
                k -= 1
                i = 0
        else:
            i += 1
    if k > 0:
        return number[:-k]
    return number
