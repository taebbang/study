name ="JAN"
answer = 0
check_a = 0
#상하이동
for i in name:
    if ord(i) <= 77:
        answer += ord(i) - 65
    else:
        answer += 1
        answer += 90-ord(i)
#좌우이동
first_A_index = 0 # 첫번째 A가 나오는 index찾기
index = 0
reverse_index = 0
num_a = 0
if "A" in name:
    #A의 갯수 찾기
    for i in name:
        if i == "A":
            num_a += 1
    #A가 아닌게 가장 처음에 나오는 곳
    for i in name:
        index += 1
        if i != "A":
            break
    # A가 가장 처음에 나오는 곳
    for i in name:
        first_A_index += 1
        if i == "A":
            break
    #반대로 A가 가장 처음 나오는 곳
    for i in reversed(name):
        reverse_index += 1
        if i != "A":
            break
    print(index)
    print(reverse_index)
    # 좌우 인덱스 위치가 같고, A가 아닌 인덱스 다음에 A가 나올경우
    if index == reverse_index and index+1 ==first_A_index:
        answer += reverse_index
    elif first_A_index <= len(name)/2:
        # answer += len(name) -first_A_index
        answer += num_a
    else:
        answer += len(name)-first_A_index
if "A" not in name:
    answer += len(name) - 1

print(answer)
