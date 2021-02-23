class Solution(object):
    def twoSum(self, numbers, target):
  
      """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        index = [0, 0]
        start =0 
        end = len(numbers)-1 
        while (start < end):
            if numbers[start] + numbers[end] > target:
                end -= 1
            elif numbers[start] + numbers[end] < target: 
                start +=1
            else:
                index[0] = start+1
                index[1] = end+1
                break
        return index
