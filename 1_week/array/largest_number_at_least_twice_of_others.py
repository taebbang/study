class Solution(object):
    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums2 = nums[:]
        nums2.sort()
        largest = nums2[len(nums)-1]
        secondary_largest = nums2[len(nums)-2]
        index = nums.index(largest)
        if len(nums) == 1 :
            return 0
        if secondary_largest*2 <= largest:
            return(index)
        else:
            return(-1)
