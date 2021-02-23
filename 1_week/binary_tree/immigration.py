def solution(n, times):
    left = 0
    right = times[len(times)-1]*n
    answer = 0
    while left <= right:
        mid = int((left + right)/2)
        num = 0
        for i in range(len(times)):
            num += int(mid/times[i])
        if num < n:
            left = mid + 1
        else:
            right = mid -1
            answer = mid
    return answer
