class Solution(object):
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        total = sum(nums)
        if len(nums) == 0:
            return -1
        if total-nums[0] == 0:
            return 0
       
        for i in range(1, len(nums)-1):
            left = left + nums[i-1]
            if left*2 == (total-nums[i]):
                return i
        if total - nums[len(nums)-1] == 0:
            return len(nums)-1
        return -1
