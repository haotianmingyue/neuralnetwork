# # 开发者 haotian
# # 开发时间: 2021/9/22 15:52
# # t = list()
# # t.append([1,2,3])
# # t.append([2])
# # print(t)
# # D = [[[0]*3]*3 for _ in range(3)]
# # print(D)
# import collections
# import random
#
# l = [0]*100000
# for i in range(100000):
#     l[i] = random.randint(0,1000)
#
# class Solution:
#     def waysToSplit(self, nums: list[int]) -> int:
#         n = len(nums)
#         if n < 3:
#             return 0
#         c : int = 0
#         for i in range(1,n):
#             nums[i] += nums[i-1]
#         for l in range(n-2):
#             for r in range(l+1,n):
#                 if nums[n-1] - nums[r-1] >= nums[r-1] - nums[l] and nums[r-1] - nums[l] >= nums[l]:
#                     c +=1
#         return c%100000007
#
# if __name__ == '__main__':
#     s = Solution()
#     print(s.waysToSplit(l))
# print(list(bin(100000)))
# print(	list('{:013b}'.format(100)))
from read_iris import read_iris_data
print(read_iris_data())