class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        slice = 0
        if len(strs) == 0:
            return ""
        if len(strs) == 1:
            return strs[0]
        for i in range(len(strs[0])):
            num = 0
            check = 0
            for j in range(len(strs)):
                if j != 0:
                    if not (strs[j].startswith((strs[0])[0:i+1])):
                        break
                    else:
                        print(i)
                        num = i +1
                        check += 1
            if check == len(strs)-1:
                slice = num
        return (strs[0])[0:slice]
