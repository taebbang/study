class Solution(object):
    def plusOne(self, digits):
        
        for i in range(1,len(digits)+1):
            if(digits[len(digits)-i] == 9):
                digits[len(digits)-i] = 0
                if digits[0] == 0:
                    digits.insert(0,1)
            else:
                digits[len(digits)-i] += 1
                break
        
    
        return digits
