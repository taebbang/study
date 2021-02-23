def solution(distance, rocks, n):
    rocks.sort()
    rocks.append(distance)
    answer = 0
    start = 0
    end = distance
    rnum = len(rocks)
    while (start <= end):
        mid = (start +end)/2
        num = 0
        mins = float("inf")
        last = 0
        for i in range(len(rocks)):
            if rocks[i] - last < mid:
                num += 1
            else:
                mins = min(mins, rocks[i]-last)
                last = rocks[i]
        if num > n:
            end = mid -1
        else:
            answer = mins
            start = mid +1
    return answer
