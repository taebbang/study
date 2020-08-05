def solution(n, lost, reserve):
    answer = n-len(lost)
    for i in lost:
        if i in reserve:
            lost.remove(i)
            reserve.remove(i)
            answer+=1
    for i in range(0,len(lost)):
        if lost[i] in reserve:
            reserve.remove(lost[i])
            answer+=1
            continue
        if (lost[i]-1) in reserve:
            reserve.remove(lost[i]-1)
            answer+=1
            continue
        elif (lost[i]+1) in reserve:
            reserve.remove(lost[i]+1)
            answer+=1
            continue
    return answer
