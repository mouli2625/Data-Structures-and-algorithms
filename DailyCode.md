i. Next Permutation:

    class Solution:
    
        def swap(self,arr,index1,index2):
            arr[index1],arr[index2]=arr[index2],arr[index1]

        def reverseArrayfromIndex(self,arr,index):
            left=index
            right=len(arr)-1
            while left<right:
                self.swap(arr,left,right)
                left+=1
                right-=1

        def nextPermutation(self, arr: List[int]) -> None:
            n=len(arr)
            index1=-1
            index2=-1
            lastIndex=n-1
            secondLastIndex=n-2
            startIndex=0
            isLastPermutation=False

            for curIndex in range(secondLastIndex,-1,-1):
                nextIndex=curIndex+1
                if arr[curIndex]<arr[nextIndex]:
                    index1=curIndex
                    break
            if index1<0:
                self.reverseArrayfromIndex(arr,0)
            else:
                for curIndex in range(lastIndex,index1,-1):
                    if arr[curIndex]>arr[index1]:
                        index2=curIndex
                        break
                self.swap(arr,index1,index2)
                self.reverseArrayfromIndex(arr,index1+1)
            print(arr)

ii. Sort 0,1,2:

    class Solution:

        def moveElementToEnd(self,arr,index):
            n=len(arr)
            element=arr[index]
            lastIndex=n-1
            for i in range(index,lastIndex):
                nextIndex=i+1
                arr[i]=arr[nextIndex]
            arr[lastIndex]=element

        def moveElementToFront(self,arr,index):
            n=len(arr)
            element=arr[index]
            startIndex=0
            for i in range(index,startIndex,-1):
                prevIndex=i-1
                arr[i]=arr[prevIndex]
            arr[startIndex]=element

        def sortColors(self, arr: List[int]) -> None:
            n=len(arr)
            left=0
            right=n-1
            cur=0
            while cur<=right:
                curElement=arr[cur]
                if curElement==0:
                    self.moveElementToFront(arr,cur)
                    left+=1
                    cur+=1
                elif curElement==1:
                    cur+=1
                else:
                    self.moveElementToEnd(arr,cur)
                    right-=1
            return arr
        
iii. Buy or sell Stock:
    
    def maxProfit(self, prices: List[int]) -> int:
        n=len(prices)
        profit=0
        minimal=sys.maxsize
        for i in range(n):
            minimal=min(minimal,prices[i])
            profit=max(profit,prices[i]-minimal)
        return profit

iv. Merge Intervals:
    
    def merge(self, arr: List[List[int]]) -> List[List[int]]:
        arr.sort(key=lambda x: x[0]) # sort based on first element
        n=len(arr)
        i=0
        while i<len(arr)-1:
            curIntervalStart=arr[i][0]
            curIntervalEnd=arr[i][1]
            nextIntervalStart=arr[i+1][0]
            nextIntervalEnd=arr[i+1][1]

            if curIntervalEnd>=nextIntervalStart:
                minIntervalStart=min(curIntervalStart,nextIntervalStart)
                maxIntervalEnd=max(curIntervalEnd,nextIntervalEnd)
                newInterval=[minIntervalStart,maxIntervalEnd]
                arr.pop(i)
                arr[i]=newInterval
                i-=1
            i+=1
        return arr

v. Longest Consecutive Sequence:
    
    def longestConsecutive(self, arr: List[int]) -> int:
        if len(arr)==0:
            return 0
        arr.sort()
        ans=[]
        res=1
        c=1
        flag=0
        for i in range(len(arr)-1):
            if arr[i]+1==arr[i+1]:
                c+=1
                flag=1
            elif arr[i]==arr[i+1]:
                pass
            elif flag:
                ans.append(c)
                c=1
                flag=0
        ans.append(c)
        return max(ans)

vi. Pascal's triangle:

    def generate(self, numRows: int) -> List[List[int]]: #print triangle
        result=[]
        for i in range(numRows):
            result.append([0]*(i+1))
            result[i][0]=1
            result[i][i]=1
            for j in range(1,i):
                result[i][j]=result[i-1][j-1]+result[i-1][j]
        print(self.printNthRow(4))
        return result
    def printNthRow(self,n):
        # result=[]
        # for i in range(n+1):
        #     num=math.factorial(n)
        #     deno=math.factorial(i)*math.factorial(n-i)
        #     result.append(num//deno) #nC(r-1)
        # return result
        result=[1,n]
        j=3
        for i in range(2,n//2+1): 
            num=result[-1]*j
            deno=j-1
            j+=1
            result.append(num//deno)
        return result+result[::-1]

vii. Minimum Number of Operations to Make Array Continuous: 
    [https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/]
    
    import bisect
    class Solution:
        def minOperations(self, arr: List[int]) -> int:
            # the logic behind this is that the given condition that max-min=n-1 stands true only if all the elements are continious/consecutive.
            n=len(arr)
            arr=list(set(arr))
            arr.sort()
            best=n
            print(arr)
            for i,start in enumerate(arr):
                # finding if consider each element as min so what should be max value of element that should be present/added so that their difference=n-1 and then finding the position where this element can be inserted in the sorted array.
                index_right=bisect.bisect_right(arr,start+n-1)-1 # O(long(n))
                best=min(best,n-(index_right-i+1))
            return best

viii. Count Inversion Pair: Pairs such that a[i]>a[j] and i<j;
    [https://www.codingninjas.com/codestudio/problems/count-inversions_615?leftPanelTab=1]

    def mergeSort(arr,temp,left,mid,right):
        invCount=0
        i=left #start index of left subtree
        j=mid  #start index of right subtree
        k=left #start index of resultant merged subarray

        while i<=mid-1 and j<=right:
            if arr[i]<=arr[j]:
                temp[k]=arr[i]
                k+=1
                i+=1
            else:
                temp[k]=arr[j]
                k+=1
                j+=1
                invCount+=(mid-i)

        while i<=mid-1:
            temp[k]=arr[i]
            k+=1
            i+=1
        while j<=right:
            temp[k]=arr[j]
            k+=1
            j+=1
        for i in range(left,right+1):
            arr[i]=temp[i]

        return invCount        


    def _mergeSort(arr,temp,left,right):
        invCount=0
        if left<right:
            mid=(left+right)//2
            invCount+=_mergeSort(arr,temp,left,mid)
            invCount+=_mergeSort(arr,temp,mid+1,right)

            invCount+=mergeSort(arr,temp,left,mid+1,right)
        return invCount

    def getInversions(arr, n) : # time:O(nlogn); space:O(n)
        n=len(arr)
        temp=[0]*n
        ans=_mergeSort(arr,temp,0,n-1)
        return ans
        
        
ix. Reverse Pairs: Count pair such that a[i]<=2*a[j] where i<j
    [https://leetcode.com/problems/reverse-pairs/]
                                                               
    class Solution:
        def mergeSort(self,arr,left,mid,right):
            count=0
            j=mid+1
            for i in range(left,mid+1):
                while j<=right and arr[i]>(2*arr[j]):
                    j+=1
                count+=(j-(mid+1))

            # usual merge algo 
            temp=[]
            i=left
            k=mid+1
            while i<=mid and k<=right:
                if arr[i]<=arr[k]:
                    temp.append(arr[i])
                    i+=1
                else:
                    temp.append(arr[k])
                    k+=1

            while i<=mid:
                temp.append(arr[i])
                i+=1
            while k<=right:
                temp.append(arr[k])
                k+=1

            for j in range(left,right+1):
                arr[j]=temp[j-left]

            return count        

        def _mergeSort(self,arr,left,right):
            if left>=right:
                return 0

            mid=(left+right)//2
            invCount=self._mergeSort(arr,left,mid)
            invCount+=self._mergeSort(arr,mid+1,right)

            invCount+=self.mergeSort(arr,left,mid,right)
            return invCount

        def reversePairs(self, arr: List[int]) -> int:
            n=len(arr)
            temp=[0]*n
            ans=self._mergeSort(arr,0,n-1)
            return ans

x. Job Sequencing Problem: Greedy [https://practice.geeksforgeeks.org/problems/job-sequencing-problem-1587115620/1#]

    def JobScheduling(self,jobs,n): # NlogN + N*M
        jobs.sort(key=lambda x:-x.profit) #sort by decreasing order of profit
        maxDeadline=0
        for j in jobs:
            maxDeadline=max(maxDeadline,j.deadline)
            
        scheduler=[-1]*maxDeadline #stores job id's
        profit=0
        count=0
        for j in jobs:
            i=j.deadline-1
            # the job with the current profit is placed at last deadline day possible.
            # so if on that day some job is already fixed so we take previous possible 
            # day for this job.
            while i>=0: 
                if scheduler[i]==-1:
                    scheduler[i]=j.id
                    profit+=j.profit
                    count+=1
                    break
                i-=1
        return [count,profit]

xi. N meetings in One room: Greedy [https://practice.geeksforgeeks.org/problems/n-meetings-in-one-room-1587115620/1#]

    def maximumMeetings(self,n,start,end): # O(NlogN)+N*M
        meetings=[]
        maxEndTime=0
        for i in range(n):
            meetings.append([start[i],end[i]])
            maxEndTime=max(maxEndTime,end[i])
        meetings.sort(key=lambda x: x[1]) # sort based on endTime
        # since we have sorted based on endtime so in the loop we will try to 
        # schedule meeting as per tis sorting so that we get greedy solution
        
        scheduler=[-1]*maxEndTime
        count=0
        
        for meeting in meetings:
            # check if withing start and endtime no other meeting held
            isFree=True
            for i in range(meeting[0]-1,meeting[1]):
                if scheduler[i]!=-1:
                    isFree=False
                    break
                
            if isFree:
                # occupy scheduler for start and endtime
                for i in range(meeting[0]-1,meeting[1]):
                    scheduler[i]=1
                count+=1
                # print("A : ",meeting)
        return count

xii.

    


===============

A. 
    Given a clock, take time as input in any format, provide a formula to calculate the angle between the minute and hour hand.
    
Sol: 
        
        """
        How to calculate the two angles with respect to 12:00? 
        The minute hand moves 360 degrees in 60 minute(or 6 degrees in one minute) and 
        hour hand moves 360 degrees in 12 hours(or 0.5 degrees in 1 minute). In h hours and m minutes,
        the minute hand would move (h*60 + m)*6 and hour hand would move (h*60 + m)*0.5. 
        """
        
    def calcAngle(h,m):
        # validate the input
        if (h < 0 or m < 0 or h > 12 or m > 60):
            print('Wrong input')
        if (h == 12):
            h = 0
        if (m == 60):
            m = 0
            h += 1;
            if(h>12):
                h = h-12; 
        # Calculate the angles moved by hour and minute hands with reference to 12:00
        hour_angle = 0.5 * (h * 60 + m)
        minute_angle = 6 * m
        # Find the difference between two angles
        angle = abs(hour_angle - minute_angle)
        # Return the smaller angle of two possible angles
        angle = min(360 - angle, angle)
        return angle

B. 
    How one would implement int Power ?
    
Sol: 
      
    """
        power() function to work for Integers Only
        Complexity = O(log(exp))
    """
    
    int power(int base, unsigned int exp){
        if (exp == 0)
            return 1;
        int temp = power(base, exp/2);
        if (exp%2 == 0)
            return temp*temp;
        else
            return base*temp*temp;
    }
    
    """
    power() function to work for negative exp and float base.
    Complexity = O(log(exp))
    """
    
    float power(float base, int exp) {
        if( exp == 0)
           return 1;
        float temp = power(base, exp/2);       
        if (exp%2 == 0)
            return temp*temp;
        else {
            if(exp > 0)
                return base*temp*temp;
            else
                return (temp*temp)/base; //negative exponent computation 
        }
    } 
        
C.  
    100 rooms in a prison. All doors are closed initially. A policeman on round i will toggle the doors
    that are in multiples of i. After 100 rounds, what doors will be open?
    
Sol: 
      
    """
        A door is toggled in ith walk if i divides door number. 
        For example, door number 45 is toggled in 1st, 3rd, 5th, 9th,15th, and 45th walk.
        The door is switched back to an initial stage for every pair of divisors. 
        For example, 45 is toggled 6 times for 3 pairs (5, 9), (15, 3), and (1, 45). 
        It looks like all doors would become closed at the end. But there are door numbers which would become open,
        for example, 16, the pair (4, 4) means only one walk. 
        Similarly all other perfect squares like 4, 9,… 
        So the answer is 1, 4, 9, 16, 25, 36, 49, 64, 81 and 100. 
        
        Note: Mean for the 100th round number of doors open = number of perfect squares from 1 to 100.
    """
    
    # An Efficient Method to count squares between a and b
    
    import math
    def CountSquares(a, b):    # Time: O(Log n) 
        return (math.floor(math.sqrt(b)) - math.ceil(math.sqrt(a)) + 1)
        
    # Function to print all the perfect squares from the given range
    
    from math import ceil, sqrt 
    def perfectSquares(l, r) : #Time : O(n)
        # Getting the very first number
        number = ceil(sqrt(l));
        # First number's square
        n2 = number * number;
        # Next number is at the difference of
        number = (number * 2) + 1;
        # While the perfect squares are from the range
        while ((n2 >= l and n2 <= r)) :
            # Print the perfect square
            print(n2, end= " ");
            # Get the next perfect square
            n2 = n2 + number;
            # Next odd number to be added
            number += 2;
            
    if __name__ == "__main__" :
        l = 2; r = 24;
        perfectSquares(l, r);

D. 
    Given int n, the total number of players and their skill-point. Distribute the players on 2 evenly balanced teams.

Sol:
    
    """
        arr=[10,  20 , 30 , 5 , 40 , 50 , 40 , 15]
        print(equal_subarr(arr))
        >>> [[10, 20, 30, 5, 40], [50, 40, 15]]
    """
    
    def equal_subarr(arr): # Time: O(n) and space: O(n).# This would create two subarrays (may be of different length) of equal sum.
        n=len(arr)
        post_sum = [0] * (n- 1) + [arr[-1]]
        for i in range(n - 2, -1, -1):
            post_sum[i] = arr[i] + post_sum[i + 1]

        prefix_sum = [arr[0]] + [0] * (n - 1)
        for i in range(1, n):
            prefix_sum[i] = prefix_sum[i - 1] + arr[i]

        for i in range(n - 1):
            if prefix_sum[i] == post_sum[i + 1]:
                return [arr[:i+1],arr[i+1:]]
        return -1
        
    ------------------
    
    # Returns split point. If not possible, then return -1.
    def findSplitPoint(arr, n) :
        # traverse array element and compute sum of whole array
        leftSum = 0
        for i in range(0, n) :
            leftSum += arr[i]

        # again traverse array and compute right sum and also check left_sum equal to right sum or not
        rightSum = 0
        for i in range(n-1, -1, -1) :
            # add current element to right_sum
            rightSum += arr[i]

            # exclude current element to the left_sum
            leftSum -= arr[i]

            if (rightSum == leftSum) :
                return i

        # if it is not possible to split array into two parts.
        return -1

    # Prints two parts after finding split point using findSplitPoint()
    def printTwoParts(arr, n) :         # Time complexity: O(n). space : O(1)
        splitPoint = findSplitPoint(arr, n)
        if (splitPoint == -1 or splitPoint == n ) :
            print ("Not Possible")
            return
        for i in range (0, n) :
            if(splitPoint == i) :
                print ("")
            print (arr[i], end = " ")        
    
    --------------
    
    """
        One check is simple. If sum of all the elements of the array is odd then it cannot be divided in two parts with equal sum.
        But, when the total sum of array is even (say, totalSum), then we have to check if we can find a subset of array
        with sum= totalSum/2. This is the challenging part. So our problem reduces to:
        Find the subset of array with sum = totalSum/2

        Dynamic Programming approach 
        In this approach we solve the problem in bottom-up fashion. Let’s create a 2-dim array
        of size (totalSum/2)*(n+1) and try to fill each entry as shown below:
    """
    
    bool twoPartsExistsDyn(int *arr, int n){
        int totalSum = 0, i;
        for(i=0; i<n; i++)
            totalSum += arr[i];
        // if sum is odd then does not exist.
        if( (totalSum & 1) != 0)
            return false;
        // Sum of each part.
        int partSum = totalSum/2;
        // latest compilers allows variable length arrays. Else declare the array on heap
        bool partArr[partSum+1][n+1];
        // initialize top row as true
        for (i = 0; i <= n; i++)
            partArr[0][i] = true;
        // initialize first column as false (except part[0][0])
        for (i = 1; i <= partSum; i++)
            partArr[i][0] = false;
        // Fill the partition table in botton up manner
        for (i = 1; i <= partSum; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                partArr[i][j] = partArr[i][j-1];
                if (i >= arr[j-1])
                    partArr[i][j] = partArr[i][j] || partArr[i - arr[j-1]][j-1];
            }
        }
        return partArr[partSum][n];
    }

E. find a missing and repeating integer from an array of N integers ranging from 1 to N.

Sol:
    """
        Method 2 (Use count array)
        Approach: 

        Create a temp array temp[] of size n with all initial values as 0. Traverse the input array arr[], and do following for each arr[i] 
        if(temp[arr[i]] == 0) temp[arr[i]] = 1;
        if(temp[arr[i]] == 1) output “arr[i]” //repeating
        Traverse temp[] and output the array element having value as 0 (This is the missing element)
        Time Complexity: O(n) and Auxiliary Space: O(n)
    """
    """
        Method 3 (Use elements as Index and mark the visited places)
        Approach: 
        Traverse the array. While traversing, use the absolute value of every element as an index and make the value at 
        this index as negative to mark it visited. If something is already marked negative then this is the repeating element. 
        To find missing, traverse the array again and look for a positive value.
    """
    
    def printTwoElements( arr, size):  O(n) and O(1)
        for i in range(size):
            if arr[abs(arr[i])-1] > 0:
                arr[abs(arr[i])-1] = -arr[abs(arr[i])-1]
            else:
                print("The repeating element is ", abs(arr[i]))
        for i in range(size):
            if arr[i]>0:
                print("and the missing element is ", i + 1)

F. 
    Given huge file having list of numbers find the largest possible sum of any k numbers.
    
Sol:
    """
    An Efficient Solution is based on the fact that sum of a subarray (or window) of size k can be obtained 
    in O(1) time using the sum of the previous subarray (or window) of size k. Except for the first subarray 
    of size k, for other subarrays, we compute the sum by removing the first element of the last window and adding 
    the last element of the current window.
    """
    
    # Returns maximum sum in a subarray of size k.
    def maxSum(arr, n, k):    # time : O(n)   space: O(1)
        # k must be smaller than n
        if (n < k):
            print("Invalid")
            return -1
        # Compute sum of first window of size k
        res = 0
        for i in range(k):
            res += arr[i]
        # Compute sums of remaining windows by removing first element of previous window and adding last element of current window.
        curr_sum = res
        for i in range(k, n):
            curr_sum += arr[i] - arr[i-k]
            res = max(res, curr_sum)
        return res

G. 
    Given a list of words, identify words which are anagrams of each other, and print them out as sets of anagrams.
    
Sol:
    # Python code to print all anagrams together
    from collections import Counter, defaultdict
    user_input = ["cat", "dog", "tac", "edoc", "god", "tacact","act", "code", "deno", "node", "ocde", "done", "catcat"]


    def solve(words: list) -> list:   # Time: O(N*M). and Space: O(N*M)
        # defaultdict will create a new list if the key is not found in the dictionary
        m = defaultdict(list)

        # loop over all the words
        for word in words:
            # Counter('cat') : counts the frequency of the characters present in a string
            # >>> Counter({'c': 1, 'a': 1, 't': 1})

            # frozenset(dict(Counter('cat')).items()) : frozenset takes an iterable object as input and makes them immutable.
            # So that hash(frozenset(Counter('cat'))) is equal to hash of other 'cat' anagrams
            # >>> frozenset({('c', 1), ('a', 1), ('t', 1)})
            m[frozenset(dict(Counter(word)).items())].append(word)
        return [v for k, v in m.items()]
    
    """
        NOTE: More efficient way is to use trie: https://www.geeksforgeeks.org/given-a-sequence-of-words-print-all-anagrams-together-set-2/
    """

H. Given a set of strings, check whether it is possible to chain all of them. Two strings can be chained iff s1[n] == s2[0] || s2[0] == s1[n].

Sol:
    
    """
    We need to check whether this graph can have a loop which goes through all the vertices, we’ll check two conditions, 
    1. Indegree and Outdegree of each vertex should be the same.
    2. The graph should be strongly connected.
    
    The first condition can be checked easily by keeping two arrays, in and out for each character. 
    For checking whether a graph is having a loop which goes through all vertices is the same as checking complete 
    directed graph is strongly connected or not because if it has a loop which goes through all vertices then we can 
    reach to any vertex from any other vertex that is, the graph will be strongly connected and the same argument can 
    be given for reverse statement also. 

    Now for checking the second condition we will just run a DFS from any character and visit all reachable vertices
    from this, now if the graph has a loop then after this one DFS all vertices should be visited, if all vertices are
    visited then we will return true otherwise false so visiting all vertices in a single DFS flags a possible ordering among strings. 
    """
    
    # Python3 code to check if cyclic order is possible among strings under given constraints
    M = 26
    # Utility method for a depth first search among vertices
    def dfs(g, u, visit):
        visit[u] = True

        for i in range(len(g[u])):
            if(not visit[g[u][i]]):
                dfs(g, g[u][i], visit)

    # Returns true if all vertices are strongly connected i.e. an be made as loop
    def isConnected(g, mark, s):
        # Initialize all vertices as not visited
        visit = [False for i in range(M)]
        # Perform a dfs from s
        dfs(g, s, visit)
        # Now loop through all characters
        for i in range(M):
            # I character is marked (i.e. it was first or last character of some string)
            # then it should be visited in last dfs (as for looping, graph should be strongly connected) 
            if(mark[i] and (not visit[i])):
                return False
        # If we reach that means  graph is connected
        return True

    # return true if an order among strings is possible
    def possibleOrderAmongString(arr, N):
        # Create an empty graph
        g = {}
        # Initialize all vertices as not marked
        mark = [False for i in range(M)]
        # Initialize indegree and outdegree of every vertex as 0.
        In = [0 for i in range(M)]
        out = [0 for i in range(M)]
        # Process all strings one by one
        for i in range(N):
            # Find first and last characters
            f = (ord(arr[i][0]) - ord('a'))
            l = (ord(arr[i][-1]) - ord('a'))
            # Mark the characters
            mark[f] = True
            mark[l] = True
            # Increase indegree and outdegree count
            In[l] += 1
            out[f] += 1
            if f not in g:
                g[f] = []
            # Add an edge in graph
            g[f].append(l)

        # If for any character indegree is not equal to outdegree then ordering is not possible
        for i in range(M):
            if(In[i] != out[i]):
                return False
        return isConnected(g, mark, ord(arr[0][0]) - ord('a'))

    arr = ["ab", "bc", "cd", "de", "ed", "da"]
    N = len(arr)
    if(possibleOrderAmongString(arr, N) == False):
        print("Ordering not possible")
    else:
        print("Ordering is possible")


    ----------------
    """
        This problem can be simplified as Find if an array of strings can be chained to form a circle.
        {"for", "geek", "rig", "kaf"} strings can be chained as "for", "rig", "geek" and "kaf"
        
        Following are detailed steps of the algorithm.
        1) Create a directed graph g with number of vertices equal to the size of alphabet. 
        We have created a graph with 26 vertices in the below program.
        2) Do following for every string in the given array of strings. 
            a) Add an edge from first character to last character of the given graph.
        3) If the created graph has eulerian circuit, then return true, else return false.
        
        Note: Eulerian Path is a path in graph that visits every edge exactly once. 
        Eulerian Circuit is an Eulerian Path which starts and ends on the same vertex. 
        A graph is said to be eulerian if it has a eulerian cycle.
    """
    # Python program to check if a given directed graph is Eulerian or not
    CHARS = 26

    # A class that represents an undirected graph
    class Graph(object):
        def __init__(self, V):
            self.V = V	 # No. of vertices
            self.adj = [[] for x in range(V)] # a dynamic array
            self.inp = [0] * V
        def addEdge(self, v, w):
            self.adj[v].append(w)
            self.inp[w]+=1

        # Method to check if this graph is Eulerian or not
        def isSC(self):
            # Mark all the vertices as not visited (For first DFS)
            visited = [False] * self.V
            # Find the first vertex with non-zero degree
            n = 0
            for n in range(self.V):
                if len(self.adj[n]) > 0:
                    break
            # Do DFS traversal starting from first non zero degree vertex.
            self.DFSUtil(n, visited)
            # If DFS traversal doesn't visit all vertices, then return false.
            for i in range(self.V):
                if len(self.adj[i]) > 0 and visited[i] == False:
                    return False
            # Create a reversed graph
            gr = self.getTranspose()
            # Mark all the vertices as not visited (For second DFS)
            for i in range(self.V):
                visited[i] = False

            # Do DFS for reversed graph starting from first vertex. Starting Vertex must be same starting point of first DFS
            gr.DFSUtil(n, visited)

            # If all vertices are not visited in second DFS, then return false
            for i in range(self.V):
                if len(self.adj[i]) > 0 and visited[i] == False:
                    return False
            return True

        # This function returns true if the directed graph has an eulerian cycle, otherwise returns false
        def isEulerianCycle(self):
            # Check if all non-zero degree vertices are connected
            if self.isSC() == False:
                return False
            # Check if in degree and out degree of every vertex is same
            for i in range(self.V):
                if len(self.adj[i]) != self.inp[i]:
                    return False
            return True

        # A recursive function to do DFS starting from v
        def DFSUtil(self, v, visited):
            # Mark the current node as visited and print it
            visited[v] = True
            # Recur for all the vertices adjacent to this vertex
            for i in range(len(self.adj[v])):
                if not visited[self.adj[v][i]]:
                    self.DFSUtil(self.adj[v][i], visited)
        # Function that returns reverse (or transpose) of this graph. This function is needed in isSC()
        def getTranspose(self):
            g = Graph(self.V)
            for v in range(self.V):
                # Recur for all the vertices adjacent to this vertex
                for i in range(len(self.adj[v])):
                    g.adj[self.adj[v][i]].append(v)
                    g.inp[v]+=1
            return g

    # This function takes an of strings and returns true if the given array of strings can be chained to form cycle
    def canBeChained(arr, n):
        # Create a graph with 'alpha' edges. i.e. 26 edges
        g = Graph(CHARS)
        # Create an edge from first character to last character of every string
        for i in range(n):
            s = arr[i]
            g.addEdge(ord(s[0])-ord('a'), ord(s[len(s)-1])-ord('a'))
        # The given array of strings can be chained if there is an eulerian cycle in the created graph
        return g.isEulerianCycle()
        
    arr1 = ["for", "geek", "rig", "kaf"]
    n1 = len(arr1)
    if canBeChained(arr1, n1):
        print ("Can be chained")
    else:
        print ("Cant be chained")
        
I. 
    Serialize and Deserialize Binary Tree

Sol:
    
    class Node:
        def __init__(self,data):
            self.data=data
            self.left=None
            self.right=None

    class Solution:
        def serialize(self, root):
            res = list()
            self._serialize(root, res)
            return ' '.join(res)
        def _serialize(self, root, res):
            if root == None:
                res.append('#')
                return
            res.append(str(root.data))
            self._serialize(root.left, res)        
            self._serialize(root.right, res)
        def deserialize(self, data):
            data = iter(data.split())
            root = self._deserialize(data)
            return root
        def _deserialize(self, data):
            val = next(data)
            if val == '#':
                return None
            root = Node(int(val))
            root.left = self._deserialize(data)
            root.right = self._deserialize(data)
            return root

        def inorder(self,root):
            if root!=None:
                self.inorder(root.left)
                print(root.data)
                self.inorder(root.right)
    if __name__ == "__main__":
        root=Node(20)
        root.left=Node(8)
        root.right=Node(22)
        root.left.left=Node(4)
        root.left.right=Node(12)
        root.left.right.left=Node(10)
        root.left.right.right=Node(14)
        obj=Solution()
        obj.serialize(root)
        print()
        root1=None
        serializedVal=obj.serialize(root)
        deserializeVal=obj.deserialize(serializedVal)
        print("serializedVal:",serializedVal)
        print("deserializedVal:",deserializeVal)
        print("inorder: ")
        obj.inorder(deserializeVal)

J. Find Common Characters from array of strings:[https://leetcode.com/problems/find-common-characters/]
Sol: 

    def commonChars(words):  #Time: O(n*m). Space: O(26+26)
        finalCount={}
        count={}
        for i in range(ord("a"),ord("z")+1):
            finalCount[chr(i)]=100
            count[chr(i)]=0
        for word in words:
            for i in range(ord("a"),ord("z")+1):
                count[chr(i)]=0
            for ch in word:
                if ch in count.keys():
                    count[ch]+=1
                else:
                    count[ch]=1
            for ch in range(ord("a"),ord("z")+1):
                if chr(ch) in count.keys():
                    finalCount[chr(ch)]=min(finalCount[chr(ch)],count[chr(ch)])
        result=[]
        temp=""
        times=0
        for ch in range(ord("a"),ord("z")+1):
            times=finalCount[chr(ch)]
            temp=chr(ch)
            while times>0 and times!=100:
                # print(temp,end=" ")
                result.append(temp)
                times-=1
        return result

    print(commonChars(["qweerty" , "weerty" , "eerty"]))    # -> ['e', 'e', 'r', 't', 'y']
    print(commonChars(["bella" , "ciao" , "espanol"]))     # -> ['a']
    print(commonChars(["aab" , "ba" , "baa"]))             # -> ['a', 'b']
    print(commonChars(["aab" , "ba" , "baa","c"]))         # -> []
    
    -----------------------------
    
    def commonChars(self, words: List[str]) -> List[str]:
        # if only one word present in array
        if len(words)==1:
            return list(words[0])
        # if two word present in array so return their common terms
        if len(words)==2:
            return list(set(words[0]).intersection(set(words[1])))
        
        # iterate and find the common characters in all the words
        ans=set(words[0]).intersection(set(words[1]))
        for i in range(2,len(words)):
            ans=set(ans).intersection(set(words[i]))
        ans=list(ans)
        print(ans)
        
        # Store each common words in dictionary as key and a list of size=total no. of words
        dic={}
        for i in ans:
            dic[i]=[0]*len(words)
        
        # Store count of each common word occurance in each word
        for i in range(len(words)):
            for j in range(len(words[i])):
                if words[i][j] in dic:
                    dic[words[i][j]][i]+=1
        print(dic)
        
        # Iterate over the dictionary keys and for each key keep adding them in final result untill their count in any words is not zero.
        res=[]
        for key,val in dic.items():
            temp=val
            while 0 not in temp:
                res.append(key)
                for i in range(len(temp)):
                    temp[i]-=1
        return res
                
K. Remove duplicates from sorted linkedlist:

    Sol:
  
        def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
            root=head
            while head!=None and head.next!=None:
                j=head.next
                while j!=None and head.val==j.val:
                    j=j.next
                    head.next=j
                head=head.next
            return root

L. Find all combinations of elements of array which sums target. Allowed same element can be taken multiple times:
[https://leetcode.com/problems/combination-sum/submissions/]
    
    Sol:
    
        def combinationSum(self,arr:List[int],target:int)->List[List[int]]:
        #time:2^t*k where t=size of arr,and k=average number of combination can be formed.
        #Space: k*x where x is the size of data structure we use to store combinations of elements
        
            ans=[]
            ds=[]
            def f(ind,sum_,ds):
                if sum_ == target:
                    result.append(ds.copy())
                    return

                if ind == len(arr) or sum_ > target:
                    return

                ds.append(arr[ind])  #pick -> target becomes target-arr[ind] or sum+arr[ind] and add the element in ds.
                f(ind, sum_ + arr[ind], ds)
                ds.pop()

                f(ind + 1, sum_, ds) #not pick  -> ind + 1

            result = []
            f(0, 0, [])            
            return result

M. Find all combinations of elements of array which sums target. Allowed same element can not be taken multiple times:
[https://leetcode.com/problems/combination-sum-ii/]
    
    Sol:
    
        def combinationSum2(self, arr: List[int], target: int) -> List[List[int]]:
            #time:2^t * k where t=size of arr, and k=average number of combination can be formed. 
            #Space: k*x where x is the size of data structure we use to store combinations of elements

            def f(nums, target, idx, ds, ans):
                if target <= 0:
                    if target == 0:
                        ans.append(ds)
                    return
                for i in range(idx, len(nums)):
                    if i > idx and nums[i] == nums[i - 1]:
                        continue
                    f(nums, target - nums[i], i + 1, ds + [nums[i]], ans)
            ans = []
            f(sorted(arr), target, 0, [], ans)
            return ret

N. Permutations of array: [https://leetcode.com/problems/permutations/submissions/]

    Sol:
        
        def permute(self, nums: List[int]) -> List[List[int]]:
            result=[]
            if len(nums)==1:
                return [nums.copy()]
            for i in range(len(nums)):
                n=nums.pop(0)
                permutations=self.permute(nums)
                for perm in permutations:
                    perm.append(n)
                result.extend(permutations)
                nums.append(n)
            return result

O. Unique Permutations of array: [https://leetcode.com/problems/permutations-ii/]
    
    Sol:
        
        def permuteUnique(self, nums: List[int]) -> List[List[int]]:
            result=[]
            if len(nums)==1:
                return [nums.copy()]
            for i in range(len(nums)):
                n=nums.pop(0)
                permutations=self.permuteUnique(nums)
                for perm in permutations:
                    perm.append(n)
                result.extend(permutations)
                nums.append(n)
            ans=[]
            for i in result:
                if i not in ans:
                    ans.append(i)
            return ans
    

P. Text Justification: Make sentences of from words from the array such that each sentence lenght == given maxLength
[https://leetcode.com/problems/text-justification/]
    
    Sol:
    
        def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
            def justifyline(wordlist,maxWidth,left=False):
                line=''
                if(len(wordlist)==1 or left):
                    line = ' '.join(wordlist)
                    line += ' '*(maxWidth - len(line))
                    return line #left justified

                betweens = len(wordlist)-1
                spaces = maxWidth - len(''.join(wordlist))
                eq_space= spaces//betweens
                rem_space = spaces % betweens
                for word in wordlist[:-1]:
                    if rem_space:
                        rem_space-=1
                        line+=word+' '*(eq_space+1)
                    else:
                        line+=word+' '*eq_space
                line+=wordlist[-1]
                return(line)


            #main
            width=0
            L = len(words)
            for i in range(L):
                width += len(words[i]) if i==0 else len(words[i])+1
                if width>maxWidth:
                    return [justifyline(words[:i],maxWidth)]+self.fullJustify(words[i:],maxWidth)
                elif i==L-1:
                    return [justifyline(words[::],maxWidth,left=True)]
                    
Q.


========================

00. Pair with given sum:
    
    Solutions:
    
        def printPairs(arr, arr_size, sum): #Time: O(n) space:O(n)
            hashmap = {}
            for i in range(0, arr_size):
                temp = sum-arr[i]
                if (temp in hashmap):
                    print (f'Pair with given sum {sum} is ({temp},{arr[i]}) at indices ({hashmap[temp]},{i})')
                hashmap[arr[i]] = i
    
2. Merge two Sorted arrays:

    Solutions:
    
        def mergeArrays(arr1, arr2, n1, n2):  #Time:O(n+m). space: O(n+m)
            arr3 = [None] * (n1 + n2)
            i,j,k = 0,0,0
            while i < n1 and j < n2:
                # Check if current element of first array is smaller than current element of second array. 
                #If yes, store first array element and increment first array index. Otherwise do same with second array
                if arr1[i] < arr2[j]:
                    arr3[k] = arr1[i]
                    k = k + 1
                    i = i + 1
                else:
                    arr3[k] = arr2[j]
                    k = k + 1
                    j = j + 1
            # Store remaining elements of first array
            while i < n1:
                arr3[k] = arr1[i];
                k = k + 1
                i = i + 1
            # Store remaining elements of second array
            while j < n2:
                arr3[k] = arr2[j];
                k = k + 1
                j = j + 1
            return arr3
        
        def merge(m, n):  #Time : O(nlogn + mlogm). space:(1)
            i = 0;
            j = 0;
            k = n - 1;
            while (i <= k and j < m):
                if (arr1[i] < arr2[j]):
                    i+=1;
                else:
                    temp = arr2[j];
                    arr2[j] = arr1[k];
                    arr1[k] = temp;
                    j += 1;
                    k -= 1;
            arr1.sort();
            arr2.sort();

1. Replace all 0's with 5: https://practice.geeksforgeeks.org/problems/replace-all-0s-with-5/1/?track=amazon-arrays&batchId=192
    Solution: 
    
        def convertFive(n:int)->int:
          s=list(str(n))
          ans=""
          for i in range(len(s)):  #time : O(k) , #space : O(2k)   k=no. of digits
              if s[i]=="0":
                  s[i]="5"
              ans+=s[i]
          return int(ans)
          
        def convertFive(n:int)->int:
          num=n
          res=0
          place=1
          if n==0:
              res+=5*place
          while n>0:              #time : O(k)  , #spcae : O(1)    k=no. of digits
              if n%10==0:
                  res+=5*place
              n//=10
              place*=10
          return res+num
          
2. Third Largest element: https://practice.geeksforgeeks.org/problems/third-largest-element/1/?track=amazon-arrays&batchId=192#
    Solution:
    
        def thirdLargest(self,a:list, n:int)->int:     #time : O(n)  , #space : O(1)
            largest,secondLargest,thirdLargest=-1,-1,-1
            if n<3:
                return -1
            largest=max(a[0],a[1],a[2])
            thirdLargest=min(a[0],a[1],a[2])
            secondLargest=a[0]+a[1]+a[2]-largest-thirdLargest
            if n==3:
                return thirdLargest
            for i in range(3,n):
                if a[i]>largest:
                    thirdLargest=secondLargest
                    secondLargest=largest
                    largest=a[i]
                elif a[i]>secondLargest:
                    thirdLargest=secondLargest
                    secondLargest=a[i]
                elif a[i]>thirdLargest:
                    thirdLargest=a[i]
            return thirdLargest

3. Largest and Second Largest which are not equal: https://practice.geeksforgeeks.org/problems/max-and-second-max/1/?track=amazon-arrays&batchId=192
    
    Solution:
    
          def largestAndSecondLargest(self, n:int, a:list)->(int,int):  #time : O(n) , #space : O(1)
              largest,secondLargest=-1,-1
              if n<2:
                  return -1
              largest=max(a[0],a[1])
              secondLargest=min(a[0],a[1])
              if n==2:
                  if largest==secondLargest:
                      return (largest,-1)
                  return (largest,secondLargest)
              ind=2
              if largest==secondLargest:
                  for i in range(2,n):
                      if a[i]>largest:
                          secondLargest=largest
                          largest=a[i]
                          ind=i
                          break
                      elif a[i]<largest:
                          secondLargest=a[i]
                          ind=i
                          break
              for i in range(ind,n):
                  if a[i]==largest:
                      pass
                  elif a[i]>largest:
                      secondLargest=largest
                      largest=a[i]
                  elif a[i]==secondLargest:
                      pass
                  elif a[i]>secondLargest:
                      secondLargest=a[i]
              if largest==secondLargest:
                  return (largest,-1)
              return (largest,secondLargest)

4. Minimum distance between two numbers: https://practice.geeksforgeeks.org/problems/minimum-distance-between-two-numbers/1/?track=amazon-arrays&batchId=192#
    
    Solution:
    
          def minDist(self, arr:list, n:int, x:int, y:int)->int:  #time: O(n+countX*countY) , #space : O(countX+countY)
              if x not in arr or y not in arr:
                  return -1
              if x==y:
                  return 0
              dic=dict()
              dic[x]=[]
              dic[y]=[]
              for i in range(n):
                  if x==arr[i]:
                      dic[x].append(i+1)
                  if y==arr[i]:
                      dic[y].append(i+1)
              a1=dic[x]
              a2=dic[y]
              mi=n
              for i in range(len(a1)):
                  for j in range(len(a2)):
                      mi=min(mi,abs(a1[i]-a2[j]))
              return mi
              
         def minDist(self, arr:list, n:int, x:int, y:int)->int:  #time: O(n) , #space : O(1)
              xi,yi=-1,-1
              mi=10000000000
              for i in range(n):
                  if arr[i]==x:
                      xi=i
                  if arr[i]==y:
                      yi=i
                  if xi!=-1 and yi!=-1:
                      mi=min(mi,abs(xi-yi))
              if xi==-1 or yi==-1:
                  return -1
              return mi

5. Max sum path in two arrays : https://practice.geeksforgeeks.org/problems/max-sum-path-in-two-arrays/1/?track=amazon-arrays&batchId=192
 
 [Given two sorted arrays A and B of size M and N respectively. Each array contains only distinct elements but may have some elements in common with the other  array. Find the maximum sum of a path from the beginning of any array to the end of any of the two arrays. We can switch from one array to another array only at the common elements.
Note: Only one repeated value is considered in the valid path sum.]

        Example 1:
        Input:
        M = 5, N = 4
        A[] = {2,3,7,10,12}
        B[] = {1,5,7,8}
        Output: 35
        Explanation: The path will be 1+5+7+10+12
        = 35.
    
    Solution:
    
        def maxSumPath(self, arr1:list, arr2:list, m:int, n:int)->int:  #time : O(m+n)  ,  #space : O(1)
            res,sum1,sum2,i,j=0,0,0,0,0
            while i<m and j<n:
                if arr1[i]<arr2[j]:
                    sum1+=arr1[i]
                    i+=1
                elif arr1[i]>arr2[j]:
                    sum2+=arr2[j]
                    j+=1
                else:
                    res+=max(sum1,sum2)+arr1[i]
                    sum1,sum2=0,0
                    i+=1
                    j+=1
            while i<m:
                sum1+=arr1[i]
                i+=1
            while j<n:
                sum2+=arr2[j]
                j+=1
            res+=max(sum1,sum2)
            return res

6. Remove duplicate elements from array: https://practice.geeksforgeeks.org/problems/remove-duplicates-in-small-prime-array/1/?track=amazon-arrays&batchId=192#

    Solution:
    
        def removeDuplicates(self, arr:list)->list:  #time : O(n)  ,  #space : O(2n)
            dic=dict()
            for i in arr:
                dic[i]=1
            ans=[]
            for i in arr:
                if dic[i]==1:
                    ans.append(i)
                    dic[i]=0
            return ans
            

7. Max sum in the configuration: https://practice.geeksforgeeks.org/problems/max-sum-in-the-configuration/1/?track=amazon-arrays&batchId=192#

    [Given an array(0-based indexing), you have to find the max sum of i*A[i] where A[i] is the element at index i in the array. The only operation allowed is to rotate(clock-wise or counter clock-wise) the array any number of times.]

        Example 1:

        Input:
        N = 4
        A[] = {8,3,1,2}
        Output: 29
        Explanation: Above the configuration
        possible by rotating elements are
        3 1 2 8 here sum is 3*0+1*1+2*2+8*3 = 29
        1 2 8 3 here sum is 1*0+2*1+8*2+3*3 = 27
        2 8 3 1 here sum is 2*0+8*1+3*2+1*3 = 17
        8 3 1 2 here sum is 8*0+3*1+1*2+2*3 = 11
        Here the max sum is 29 
    
    Solution:
    
        def max_sum(a:list,n:int)->int:   #time : O(n)  ,  #space : O(1)
            cum_sum=sum(a)
            cur_val=0
            for i in range(n):
                cur_val+=a[i]*i
            res=cur_val
            for i in range(1,n):
                next_val=cur_val-(cum_sum-arr[i-1])+arr[i-1]*(n-1)
                res=max(res,next_val)
                cur_val=next_val
            return res
        
8. Sorted subarray of 3 : https://practice.geeksforgeeks.org/problems/sorted-subsequence-of-size-3/1/?track=amazon-arrays&batchId=192

    Solution:
    
            vector<int> find3Numbers(vector<int> arr, int n) {

             int rightmax[n];
            rightmax[n-1]=arr[n-1];       
            for(int i=n-2;i>=0;i--)
            {
                rightmax[i]=max(arr[i],rightmax[i+1]);
            }

            int leftmin=INT_MAX;
            vector<int> res={};
            for(int i=0;i<=n-1;i++)
            {
                leftmin=min(arr[i],leftmin);
                 if(leftmin<arr[i] && arr[i]<rightmax[i]){
                     res={leftmin,arr[i],rightmax[i]};
                     break;
                 }

            }
            return res;
        }
9. Duplicates in array: https://practice.geeksforgeeks.org/problems/find-duplicates-in-an-array/1/?track=amazon-arrays&batchId=192#

     [Given an array a[] of size N which contains elements from 0 to N-1, you need to find all the elements occurring more than once in the given array.]
    
    Input:
    N = 5
    a[] = {2,3,1,2,3}
    Output: 2 3 
    Explanation: 2 and 3 occur more than once
    in the given array.
    
    Solution:
    
        def duplicates(self, arr:list, n:int)->int:  #time : O(n^2 + nlogn)  ,  #space : O(2n)
            dic=dict()
            for i in arr:
                if i in dic:
                    dic[i]+=1
                else:
                    dic[i]=1
            ans=[]
            for i in dic.keys():
                if dic[i]>1:
                    ans.append(i)
            if len(ans)!=0:
                return sorted(ans)
            return [-1]

        def duplicates(self, arr:list, n:int)->int:  #time : O(n)  ,  #space : O(n)
            ans=[]
            for i in range(n):
                arr[arr[i]%n]+=n
            for i in range(n):
                if arr[i]//n>1:
                    ans.append(i)
            if len(ans)==0:
                ans.append(-1)
            return ans
            
10. Wave array: https://practice.geeksforgeeks.org/problems/wave-array-1587115621/1/?track=amazon-arrays&batchId=192#
    
    Solution:
    
        def convertToWave(self,arr:list,n:int)->list: #time : O(n)  #space : O(1)
            for i in range(0,n-1,2):
                arr[i],arr[i+1]=arr[i+1],arr[i]
            return arr

11. Mountain Subarray Problem :  https://practice.geeksforgeeks.org/problems/mountain-subarray-problem/1/?track=amazon-arrays&batchId=192#

    [We are given an array of integers and a range, we need to find whether the subarray which falls in this range has values in the form of a mountain or not. All values of the subarray are said to be in the form of a mountain if either all values are increasing or decreasing or first increasing and then decreasing. More formally a subarray [a1, a2, a3 … aN] is said to be in form of a mountain if there exists an integer K, 1 <= K <= N such that,
    a1 <= a2 <= a3 .. <= aK >= a(K+1) >= a(K+2) …. >= aN
    You have to process Q queries. In each query you are given two integer L and R, denoting starting and last index of the subarrays respectively.]


        Example 1:

        Input:
        N = 8
        a[] = {2,3,2,4,4,6,3,2}
        Q = 2
        Queries = (0,2), (1,3)
        Output:
        Yes
        No

    Solution:
    
        def processqueries(self,arr:list,n:int,m:int,q:list)->list:  #time : O(n)  ,  #space : O(2n)
            ans=[]
            left=[0]*n
            leftIncrPtr=0
            for i in range(1,n):
                if arr[i]>arr[i-1]:
                    leftIncrPtr=i
                left[i]=leftIncrPtr
                
            right=[0]*n
            right[n-1]=n-1
            rightDecrPtr=n-1
            for i in range(n-2,-1,-1):
                if arr[i]>arr[i+1]:
                    rightDecrPtr=i
                right[i]=rightDecrPtr

            for query in q:
                l,r=query
                if (right[l] >= left[r]):
                    ans.append("Yes")
                else:
                    ans.append("No")
            return ans

12. Equilibrium Point : https://practice.geeksforgeeks.org/problems/equilibrium-point-1587115620/1/?track=amazon-arrays&batchId=192

    [Given an array A of n positive numbers. The task is to find the first Equilibium Point in the array. 
    Equilibrium Point in an array is a position such that the sum of elements before it is equal to the sum of elements after it.]

        Example 1:

        Input: 
        n = 5 
        A[] = {1,3,5,2,2} 
        Output: 3 
        Explanation: For second test case 
        equilibrium point is at position 3 
        as elements before it (1+3) = 
        elements after it (2+2). 

    Solution:
    
        def equilibriumPoint(self,arr:list, n:int)->int:  #time : O(n)  ,  #space : O(1)
            s=sum(arr)
            leftSum=0
            for i,val in enumerate(arr):
                s-=val
                if leftSum==s:
                    return i+1
                leftSum+=val
            return -1
            
        def equilibriumPoint(self,arr:list, n:int)->int:  #time : O(n)  ,  #space : O(2n)
            leftSum=[0]*n
            rightSum=[0]*n
            leftSum[0]=arr[0]
            rightSum[0]=arr[n-1]

            for i in range(1,n):
                leftSum[i]=leftSum[i-1]+arr[i]  #prefix sum from left to right
                rightSum[i]=rightSum[i-1]+arr[n-1-i]  #prefix sum from right to left
            for i in range(n):
                if leftSum[i]==rightSum[n-1-i]:
                    return i+1
            return -1

13. Stickler Thief : https://practice.geeksforgeeks.org/problems/stickler-theif-1587115621/1/?track=amazon-arrays&batchId=192#

    [Stickler the thief wants to loot money from a society having n houses in a single line. He is a weird person and follows a certain rule when looting the houses. According to the rule, he will never loot two consecutive houses. At the same time, he wants to maximize the amount he loots. The thief knows which house has what amount of money but is unable to come up with an optimal looting strategy. He asks for your help to find the maximum money he can get if he strictly follows the rule. Each house has a[i]amount of money present in it.]


        Example 1:

        Input:
        n = 6
        a[] = {5,5,10,100,10,5}
        Output: 110
        Explanation: 5+100+5=110

    Solution:
    
        int FindMaxSum(int arr[], int n)
        {
            int sum1 = arr[0];
            int sum2 = 0;
            int result;
            for (int i = 1; i < n; i++){
                result = (sum1 > sum2) ? sum1 : sum2;
                sum1 = sum2 + arr[i];
                sum2 = result;
            }
            return ((sum1 > sum2) ? sum1 : sum2);
        }

14. Product Array puzzle : https://practice.geeksforgeeks.org/problems/product-array-puzzle4525/1/?track=amazon-arrays&batchId=192#

    Solution:
    
        def productExceptSelf(self, nums:list, n:int)->list: #time : O(n)  ,  #space : O(2n)
            if n==1:
                return [1]
            left=[0]*n
            right=[0]*n
            left[0]=1
            right[n-1]=1
            for i in range(1,n):
                left[i]=arr[i-1]*left[i-1]
            for i in range(n-2,-1,-1):
                right[i]=arr[i+1]*right[i+1]
            prod=[0]*n
            for i in range(n):
                prod[i]=left[i]*right[i]
            return prod

        def productExceptSelf(self, nums:list, n:int)->list: #time : O(n)  ,  #space : O(n)
            if n==1:
                return [1]
            prod=[1]*n
            i,temp=1,1
            for i in range(n):
                prod[i]=temp
                temp*=arr[i]
            temp=1
            for i in range(n-1,-1,-1):
                prod[i]*=temp
                temp*=arr[i]
            return prod

        def productExceptSelf(self, a:list, n:int)->list: #time : O(n)  ,  #space : O(1)
            prod = 1
            flag = 0
            for i in range(n):
                # Counting number of elements which have value 0
                if (a[i] == 0):
                    flag += 1
                else:
                    prod *= a[i]
            arr = [0 for i in range(n)]
            for i in range(n):
                # If number of elements in array with value 0 is more than 1 than each value in new array will be equal to 0
                if (flag > 1):
                    arr[i] = 0
                # If no element having value 0 than we will insert product/a[i] in new array
                elif (flag == 0):
                    arr[i] = (prod // a[i])
                # If 1 element of array having value 0 than all the elements except that index value , will be equal to 0
                elif (flag == 1 and a[i] != 0):
                    arr[i] = 0
                # If(flag == 1 && a[i] == 0)
                else:
                    arr[i] = prod
            return arr

15. Subarray with Given Sum: https://practice.geeksforgeeks.org/problems/subarray-with-given-sum-1587115621/1/?track=amazon-arrays&batchId=192#
    
    *Based on Sliding window concept. If Current window's sum>s so remove start element, elif sum<s so add current element else sum=s then return.
    Solution:
        
        def subArraySum(self,arr, n, s):  #O(n^2) and O(1)
            curSum=0
            for i in range(n):
                curSum=arr[i]
                for j in range(i+1,n):
                    curSum+=arr[j]
                    if curSum==s:
                        return [i+1,j+1]
            return -1
         def subArraySum(self,arr, n, s): # O(n) and O(1)
            curr_sum = arr[0]
            start = 0
            i = 1
            while i <= n:
                while curr_sum > s and start < i-1:
                    curr_sum -= arr[start]
                    start += 1
                if curr_sum == s:
                    return[start+1, i]
                if i < n:
                    curr_sum+= arr[i]
                i += 1
            return [-1]
        
16. Kadane's Algorithm: Subarray with max Contiguous sum: https://practice.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1/?track=amazon-arrays&batchId=192#
    Solution:
        
        def maxSubArraySum(self,a,size):   #O(n)
            max_so_far = -1000000000000 - 1
            max_ending_here = 0
            for i in range(0, size):
                max_ending_here = max_ending_here + a[i]
                if (max_so_far < max_ending_here):
                    max_so_far = max_ending_here
                if max_ending_here < 0:
                    max_ending_here = 0  
            return max_so_far
        def maxSubArraySum(self,a,size):   #O(n)
            max_so_far = a[0]
            max_ending_here = 0
            for i in range(0, size):
                max_ending_here = max_ending_here + a[i]
                if max_ending_here < 0:
                    max_ending_here = 0
                # Do not compare for all elements. Compare only  
                # when  max_ending_here > 0
                elif (max_so_far < max_ending_here):
                    max_so_far = max_ending_here
            return max_so_far
        def maxSubArraySum(self,a,size):   #O(n)
            max_so_far =a[0]
            curr_max = a[0]
            for i in range(1,size):
                curr_max = max(a[i], curr_max + a[i])
                max_so_far = max(max_so_far,curr_max)
            return max_so_far

17. Counter elements in two arrays: https://practice.geeksforgeeks.org/problems/counting-elements-in-two-arrays/1#
    [Given two unsorted arrays arr1[] and arr2[]. They may contain duplicates. For each element in arr1[] count elements less than or equal to it in array arr2[].]
    
    Solution:
    
        def countEleLessThanOrEqual(self,arr1,n1,arr2,n2): #O(nlogn)
            def binarySearch(x,arr,n):
                start=0
                end=n-1
                while start<=end:
                    mid=(start+end)//2
                    if arr[mid]<=x:
                        start=mid+1
                    else:
                        end=mid-1
                return end

            ans=[]
            arr2.sort()
            for i in range(n1):
                ans.append(binarySearch(arr1[i],arr2,n2)+1)
            return ans

18. Count Number of possible triangles: https://practice.geeksforgeeks.org/problems/count-possible-triangles-1587115620/1/?track=amazon-sorting&batchId=192#
    [Given an unsorted array arr[] of n positive integers. Find the number of triangles that can be formed with three different array elements as lengths of three sides of triangles. ]
   
        Example:
            n = 5
            arr[] = {6, 4, 9, 7, 8}
            Output: 
            10
            Explanation: 
            There are 10 triangles
            possible  with the given elements like
            (6,4,9), (6,7,8),...
        Solution : 
            def findNumberOfTriangles(self, arr, n): #O(nlogn)   for each element from back count #pairs whose sum>current element
                arr.sort()
                c=0
                for i in range(n-1,1,-1):
                    low,high=0,i-1
                    while low<high:
                        if arr[low]+arr[high]>arr[i]:
                            c+=high-low
                            high-=1
                        else:
                            low+=1
                return c

20. find Top K frequent Elements: https://practice.geeksforgeeks.org/problems/top-k-frequent-elements-in-array/1/?track=amazon-sorting&batchId=192#
    [Given a non-empty array of integers, find the top k elements which have the highest frequency in the array. If two numbers have the same frequency then the larger number should be given preference.]
    
        Example:
            Input:
                N = 8
                nums = {1,1,2,2,3,3,3,4}
                k = 2
                Output: {3, 2}
                Explanation: Elements 1 and 2 have the
                same frequency ie. 2. Therefore, in this
                case, the answer includes the element 2
                as 2 > 1.
                
        Solutions:
            def topK(self, arr, k):  #O(dlogd) where d is frequency of elements and space : O(d)
                d = defaultdict(lambda: 0)
                for i in range(len(arr)):
                    d[arr[i]] += 1
                arr.sort(key=lambda x: (d[x], x)) # sorting array based on frequency &&  on value 
                ans=[]
                for i in arr[::-1]:
                    if i not in ans and k>0:
                        ans.append(i)
                        k-=1
                return ans
           
           def topK(self, arr, k):  #O(klogd+d) where d is frequency of elements and space : O(d); here creating a priority queue of max heap as finding max element in max heap takes log(n) time.
           import heapq
           
                ans=[]
                d = defaultdict(lambda: 0)
                for i in range(len(arr)):
                    d[arr[i]] += 1

                # Using heapq data structure
                heap = [(value, key) for key, 
                         value in d.items()]  #list of pairs (frequency,element). <- priority queue
                # print("heap : ",heap)

                # Get the top k elements
                largest = heapq.nlargest(k, heap)  # O( k log d) taking k largest elements from max heap
                # print("largestK : ",largest)

                # Print the top k elements
                for i in range(k):
                    # print(largest[i][1], end =" ")
                    ans.append(largest[i][1])

                # print(ans)
                return ans
         [referrence : https://www.geeksforgeeks.org/find-k-numbers-occurrences-given-array/]
       
21. Mid element of LinkedList: 
    [https://practice.geeksforgeeks.org/problems/finding-middle-element-in-a-linked-list/1/?track=amazon-linkedlists&batchId=192]
    
    Solution: *also could have been done by finding length of list and then return len/2 th element from list as done on next problem.
    
        def findMid(self, head): # time : O(n) space: O(1)
            midNode=head.   #for first element mid is first itself
            c=1
            if head.next!=None:    #for 2 elements mid is the second element
                head=head.next
                c+=1
                midNode=midNode.next
            if head.next!=None:     #for 3 elements mid is the second element
                head=head.next
                c+=1
            while head.next!=None:    #if counter of elements is even so mid = mid.next else pass
                head=head.next
                c+=1
                if c%2==0:
                    midNode=midNode.next
                else:
                    pass
            return midNode.data

22. Delete Mid Element of linkedList:
    [https://practice.geeksforgeeks.org/problems/delete-middle-of-linked-list/1/?track=amazon-linkedlists&batchId=192#]
    
    Solution:
        
        def deleteMid(root):
            '''
            head:  head of given linkedList
            return: head of resultant llist
            '''
            head=root
            if root==None or root.next==None:
                return None
            c=0
            while head!=None:    
                head=head.next
                c+=1
            temp=root
            for i in range(c//2-1):
                temp=temp.next
            temp.next=temp.next.next
            return root

23. Add One to number represented as linkedList: 
    [https://practice.geeksforgeeks.org/problems/add-1-to-a-number-represented-as-linked-list/1/?track=amazon-linkedlists&batchId=192#]
    
    Solution:
    
        def addOne(self,head):
            def reverse(head):
                prev=None
                cur=head
                while cur!=None:
                    temp=cur.next
                    cur.next=prev
                    prev=cur
                    cur=temp
                return prev
            head1=reverse(head)
            prev=None
            cur=head1
            carry=1
            while head1!=None:
                sum=head1.data+carry
                carry=sum//10
                sum=sum%10
                head1.data=sum
                prev=head1
                head1=head1.next
            if carry>0:
                prev.next=Node(carry)
            res=reverse(cur)
            return res

24. Detect loop in LinkedList: 
    [https://practice.geeksforgeeks.org/problems/detect-loop-in-linked-list/1/?track=amazon-linkedlists&batchId=192#]
    
    Solution:
    
        def detectLoop(self, head):
            fast_ptr, slow_ptr = head, head
            while fast_ptr and slow_ptr and fast_ptr.next:
                fast_ptr, slow_ptr = fast_ptr.next.next, slow_ptr.next
                if fast_ptr is slow_ptr:
                    return True
            return False

25. Count Nodes in Loop of linkedList or the length of loof in the linkedList:
    [https://practice.geeksforgeeks.org/problems/find-length-of-loop/1/?track=amazon-linkedlists&batchId=192#]
    
    Solution:
    
        def countNodesinLoop(head):
            def detectLoop(head):
                fast_ptr, slow_ptr = head, head
                while fast_ptr and slow_ptr and fast_ptr.next:
                    fast_ptr, slow_ptr = fast_ptr.next.next, slow_ptr.next
                    if fast_ptr is slow_ptr:
                        return fast_ptr
                return False
            start = detectLoop(head)
            c=1
            cur=start
            if start:
                while start.next!=cur:
                    c+=1
                    start=start.next
                return c
            else:
                return 0
26. Remove Loop from linkedList:
    [https://practice.geeksforgeeks.org/problems/remove-loop-in-linked-list/1/?track=amazon-linkedlists&batchId=192#]
    
    Solution:
    
        def removeLoop(self, head):
            fast_ptr, slow_ptr = head, head
            while fast_ptr!=None and fast_ptr.next!=None:
                fast_ptr, slow_ptr = fast_ptr.next.next, slow_ptr.next
                if fast_ptr is slow_ptr:
                    break
            if slow_ptr!=fast_ptr:
                return
            if slow_ptr is head:
                while fast_ptr.next!=slow_ptr:
                    fast_ptr=fast_ptr.next
                fast_ptr.next=None
                return
            slow_ptr=head
            while fast_ptr.next!=slow_ptr.next:
                fast_ptr=fast_ptr.next
                slow_ptr=slow_ptr.next
            fast_ptr.next=None

27. Merge two Sorted linkedlists:
    []
    
    Solutions:
    
        def sortedMerge(headA, headB):
            # A dummy node to store the result
            dummyNode = Node(0)
            # Tail stores the last node
            tail = dummyNode
            while True:
                # If any of the list gets completely empty
                # directly join all the elements of the other list
                if headA is None:
                    tail.next = headB
                    break
                if headB is None:
                    tail.next = headA
                    break
                # Compare the data of the lists and whichever is smaller is
                # appended to the last of the merged list and the head is changed
                if headA.data <= headB.data:
                    tail.next = headA
                    headA = headA.next
                else:
                    tail.next = headB
                    headB = headB.next
                # Advance the tail
                tail = tail.next
            # Returns the head of the merged list
            return dummyNode.next

28. Merge k sorted linkedlists:
    []
    
    Solutions:
    
        import heapq
        def mergeKLists(self,arr,K): #if n is the maximum size of all the k link list then  #Time: O(k*n + nlogn).  #space: O(n+n)
            pq=[]
            for subarr in arr:
                temp=subarr
                while temp!=None:
                    pq.append(temp.data)
                    temp=temp.next
            brr = heapq.nlargest(len(pq), pq)
            root=Node(0)
            temp=root
            while len(brr)!=0:
                temp.next=Node(brr.pop())
                temp=temp.next
            temp.next=None
            return root.next
       
        def mergeKLists(self,arr,k):  #Time: O(N*log k) or O(n*k*log k). #Space: O(N) or O(n*k)
            k-=1
            # Takes two lists sorted in increasing order,
            # and merge their nodes together to make one
            # big sorted list. Below function takes
            # O(Log n) extra space for recursive calls,
            # but it can be easily modified to work with
            # same time and O(1) extra space
            def SortedMerge(a, b):
                result = None
                # Base cases
                if (a == None):
                    return(b)
                elif (b == None):
                    return(a)
                # Pick either a or b, and recur
                if (a.data <= b.data):
                    result = a
                    result.next = SortedMerge(a.next, b)
                else:
                    result = b
                    result.next = SortedMerge(a, b.next)
                return result
            # Repeat until only one list is left
            while (k != 0):
                i = 0
                j = k
                # (i, j) forms a pair
                while (i < j):
                    # Merge List i with List j and store
                    # merged list in List i
                    arr[i] = SortedMerge(arr[i], arr[j])
                    # Consider next pair
                    i += 1
                    j -= 1
                    # If all pairs are merged, update last
                    if (i >= j):
                        k = j
            return arr[0]
    
        def mergeKArrays(self, arr, k):  #time: O(K2*Log(K)). space : O(k)
            brr=[]
            for i in arr:
                brr+=i
            return heapq.nlargest(len(brr),brr)[::-1]   # or heapq.nsmallest(len(brr),brr)
        
29. Maximum of minimum for every window size.
    [https://practice.geeksforgeeks.org/problems/maximum-of-minimum-for-every-window-size3453/1/?track=amazon-stacks&batchId=192#]
    
    Solutions:
    
        def maxOfMin(self,arr,n): # time : O(n) and space : O(n)
            s = [] # Used to find previous and next smaller
            # Arrays to store previous and next smaller. Initialize elements of left[] and right[]
            left = [-1] * (n + 1)
            right = [n] * (n + 1)
            for i in range(n):
                while (len(s) != 0 and arr[s[-1]] >= arr[i]):
                    s.pop()
                if (len(s) != 0):
                    left[i] = s[-1]
                s.append(i)
            # Empty the stack as stack is going to be used for right[]
            while (len(s) != 0):
                s.pop()
            for i in range(n - 1, -1, -1):
                while (len(s) != 0 and arr[s[-1]] >= arr[i]):
                    s.pop()
                if(len(s) != 0):
                    right[i] = s[-1]
                s.append(i)
            ans = [0] * (n + 1)
            for i in range(n + 1):
                ans[i] = 0
            for i in range(n):
                # Length of the interval
                Len = right[i] - left[i] - 1
                # arr[i] is a possible answer for this Length 'Len' interval, check if arr[i] is more than max for 'Len'
                ans[Len] = max(ans[Len], arr[i])
            # Some entries in ans[] may not be filled yet. Fill them by taking values from right side of ans[]
            for i in range(n - 1, 0, -1):
                ans[i] = max(ans[i], ans[i + 1])
            return ans[1:]

30. Queue using two stacks:
    [https://practice.geeksforgeeks.org/problems/queue-using-two-stacks/1/?track=amazon-queue&batchId=192#]
    
    Solutions:
    
        #Function to push an element in queue by using 2 stacks.
        def Push(x,stack1,stack2):
            '''
            x: value to push
            stack1: list.  
            stack2: list.  
            '''
            while len(stack1)>0:
                stack2.append(stack1.pop())
            stack1.append(x)
            while len(stack2)>0:
                stack1.append(stack2.pop())

        #Function to pop an element from queue by using 2 stacks.
        def Pop(stack1,stack2):

            '''
            stack1: list
            stack2: list
            '''
            ans=-1
            if len(stack1)>0:
                ans = stack1.pop()
            return ans

31. LRU Cache: 

    Design a data structure that works like a LRU Cache. Here cap denotes the capacity of the cache and Q denotes the number of queries. Query can be of two
    types:
    
    a) SET x y : sets the value of the key x with value y.
    
    b) GET x : gets the key of x if present else returns -1.
    
    The LRUCache class has two methods get() and set() which are defined as follows.
    
    a) get(key)   : returns the value of the key if it already exists in the cache otherwise returns -1.
    
    b) set(key, value) : if the key is already present, update its value. If not present, add the key-value pair to the cache. If the cache reaches its capacity it should invalidate the least recently used item before inserting the new item.
    
    In the constructor of the class the capacity of the cache should be intitialized.

    [https://practice.geeksforgeeks.org/problems/lru-cache/1/?track=amazon-queue&batchId=192#]
    
    Example:

        Input:
        cap = 2
        Q = 8
        Queries = SET 1 2 SET 2 3 SET 1 5
        SET 4 5 SET 6 7 GET 4 SET 1 2 GET 3
        Output: 5 -1
        Explanation: 
        Cache Size = 2
        SET 1 2 : 1 -> 2

        SET 2 3 : 1 -> 2, 2 -> 3 (the most recently 
        used one is kept at the rightmost position) 

        SET 1 5 : 2 -> 3, 1 -> 5

        SET 4 5 : 1 -> 5, 4 -> 5 (Cache size is 2, hence 
        we delete the least recently used key-value pair)

        SET 6 7 : 4 -> 5, 6 -> 7 

        GET 4 : Prints 5 (The cache now looks like
        6 -> 7, 4->5)

        SET 1 2 : 4 -> 5, 1 -> 2 
        (Cache size is 2, hence we delete the least 
        recently used key-value pair)

        GET 3 : No key value pair having 
        key = 3. Hence, -1 is printed.

    
    Solutions:
        
        from collections import OrderedDict
        class LRUCache:
            #Constructor for initializing the cache capacity with the given value.  
            def __init__(self,cap):
                self.cap=cap
                self.cache=OrderedDict() # dictionary that maintains order of inserted key-values

            #Function to return value corresponding to the key.
            def get(self, key):
                if key in self.cache:
                    self.cache.move_to_end(key,last=True) #since recently used so move to last
                    return self.cache[key]
                else:
                    return -1

            #Function for storing key-value pair.   
            def set(self, key, value):
                if key in self.cache:
                    self.cache[key]=value     #update existing key
                    self.cache.move_to_end(key,last=True) #move updated key to end because updating a key value don't affect the order of ordered dict
                else:
                    self.cache[key]=value       #add new key at the last 
                    if len(self.cache)>self.cap:
                        self.cache.popitem(last=False) #remove key from begining

32. Maximum of all subarrays of size k:
    [https://practice.geeksforgeeks.org/problems/maximum-of-all-subarrays-of-size-k3101/1/?track=amazon-queue&batchId=192#]
    
    Example: 
    
        Input:
        N = 9, K = 3
        arr[] = 1 2 3 1 4 5 2 3 6
        Output: 
        3 3 4 5 5 5 6 
        Explanation: 
        1st contiguous subarray = {1 2 3} Max = 3
        2nd contiguous subarray = {2 3 1} Max = 3
        3rd contiguous subarray = {3 1 4} Max = 4
        4th contiguous subarray = {1 4 5} Max = 5
        5th contiguous subarray = {4 5 2} Max = 5
        6th contiguous subarray = {5 2 3} Max = 5
        7th contiguous subarray = {2 3 6} Max = 6
    
    Solution:
    
        def max_of_subarrays(self,arr,n,k): #time O(n), space O(k)
            ans=[]
            ans.append(max(arr[:k]))
            j=0
            for i in range(k,n):
                if arr[j]==ans[-1]:
                    ans.append(max(arr[j+1:i+1]))
                elif arr[i]>ans[-1]:
                    ans.append(arr[i])
                else:
                    ans.append(ans[-1])
                j+=1

            return ans

33. Tree Traversal using Iterative approach to return list of nodes in traversal:

    InOrder:
    
        from collections import deque
        def InOrder(self,root):
            ans=[]
            # create an empty stack
            stack = deque()
            # start from the root node (set current node to the root node)
            curr = root
            # if the current node is None and the stack is also empty, we are done
            while stack or curr:
                # if the current node exists, push it into the stack (defer it)
                # and move to its left child
                if curr:
                    stack.append(curr)
                    curr = curr.left
                else:
                    # otherwise, if the current node is None, pop an element from the stack,
                    # print it, and finally set the current node to its right child
                    curr = stack.pop()
                    ans.append(curr.data)
                    # print(curr.data, end=' ')
                    curr = curr.right
            return ans
    
    PreOrder:
    
         # Iterative function to perform preorder traversal on the tree
        def preorderIterative(root):
            ans=[]
            # return if the tree is empty
            if root is None:
                return
            # create an empty stack and push the root node
            stack = deque()
            stack.append(root)
            # start from the root node (set current node to the root node)
            curr = root
            # loop till stack is empty
            while stack:
                # if the current node exists, print it and push its right child
                # to the stack before moving to its left child
                if curr:
                    ans.append(curr.data)
                    # print(curr.data, end=' ')
                    if curr.right:
                        stack.append(curr.right)
                    curr = curr.left
                # if the current node is None, pop a node from the stack
                # set the current node to the popped node
                else:
                    curr = stack.pop()
           return ans
    
    PostOrder:
    
        # Iterative function to perform postorder traversal on the tree
        def postorderIterative(root):
            ans=[]
            # return if the tree is empty
            if root is None:
                return
            # create an empty stack and push the root node
            stack = deque()
            stack.append(root)
            # create another stack to store postorder traversal
            ## out = deque()
            # loop till stack is empty
            while stack:
                # pop a node from the stack and push the data into the output stack
                curr = stack.pop()
                ans.append(curr.data)
                ## out.append(curr.data)
                # push the left and right child of the popped node into the stack
                if curr.left:
                    stack.append(curr.left)
                if curr.right:
                    stack.append(curr.right)
           return ans
            # print postorder traversal
            # while out:
            #   print(out.pop(), end=' ')
            
34. Construct tree from inorder and preorder:
    [https://practice.geeksforgeeks.org/problems/construct-tree-1/1/?track=amazon-trees&batchId=192#]
    
    Solution:
    
        def buildtree(self, inorder, preorder, n):  #time: O(n) and spcae: O(n)
            def array_to_tree(left, right):
                nonlocal preorder_index
                # if there are no elements to construct the tree
                if left > right: return None
                # select the preorder_index element as the root and increment it
                root_value = preorder[preorder_index]
                root = Node(root_value)
                preorder_index += 1
                # build left and right subtree
                # excluding inorder_index_map[root_value] element because it's the root
                root.left = array_to_tree(left, inorder_index_map[root_value] - 1)
                root.right = array_to_tree(inorder_index_map[root_value] + 1, right)
                return root
            preorder_index = 0
            # build a hashmap to store value -> its index relations
            inorder_index_map = {}
            for index, value in enumerate(inorder):
                inorder_index_map[value] = index
            return array_to_tree(0, len(preorder) - 1)

35. Right view of tree:
    
    Solution:
        
        def rightView(self,root): #Time: O(n) space: O(n)
            def rightViewUtil(root, level, max_level):
                nonlocal ans
                # Base Case
                if root is None:
                    return
                # If this is the last node of its level
                if (max_level[0] < level):
                    ans.append(root.data)
                    max_level[0] = level
                # Recur for right subtree first, then left subtree
                rightViewUtil(root.right, level+1, max_level)
                rightViewUtil(root.left, level+1, max_level)
            ans=[]
            max_level = [0]
            rightViewUtil(root, 1, max_level)
            return ans

36. K distance from root:
    [https://practice.geeksforgeeks.org/problems/k-distance-from-root/1/?track=amazon-trees&batchId=192#]
    
    Solutions:
    
        def KDistance(root,k):
            def levelOrder2(root,level):#level-wise traversal
                nonlocal ans
                if root==None:
                    return root
                if level==1:
                    ans.append(root.data)
                elif level>1:
                    levelOrder2(root.left,level-1)
                    levelOrder2(root.right,level-1)
            ans=[]
            levelOrder2(root,k+1)
            return ans

37. Mirror tree:
    [https://practice.geeksforgeeks.org/problems/mirror-tree/1/?track=amazon-trees&batchId=192#]
    
    Solutions:
    
        def mirror(self,node):  #Time: O(n) space: O(h)
            if (node == None):
                return
            else:
                temp = node
                """ do the subtrees """
                self.mirror(node.left)
                self.mirror(node.right)
                """ swap the pointers in this node """
                temp = node.left
                node.left = node.right
                node.right = temp

38. Maximum width of tree:
    []
    
    Solutions:
        
        def getMaxWidth(self,root):  #time : O(n) space: O(n)
            def height(node):
                if node is None:
                    return 0
                else:
                    lHeight = height(node.left)
                    rHeight = height(node.right)
                    return (lHeight+1) if (lHeight > rHeight) else (rHeight+1)

            def getMaxWidthRecur(root, count, level):
                if root is not None:
                    count[level] += 1
                    getMaxWidthRecur(root.left, count, level+1)
                    getMaxWidthRecur(root.right, count, level+1)
            def getMax(count, n):
                max = count[0]
                for i in range(1, n):
                    if (count[i] > max):
                        max = count[i]
                return max

            h = height(root)
            # Create an array that will store count of nodes at each level
            count = [0] * h
            level = 0
            # Fill the count array using preorder traversal
            getMaxWidthRecur(root, count, level)
            # Return the maximum value from count array
            return getMax(count, h)
           
39. Tree:

    Solutions:
        
        class Node:
            def __init__(self, data):
                self.left=None
                self.right=None
                self.data=data
        class Tree: 
            def preOrder(self, root:Node):
                if root is not None:
                    print(root.data,end= " ")
                    self.preOrder(root.left)
                    self.preOrder(root.right)
            def inOrder(self, root:Node):
                if root is not None:
                    self.inOrder(root.left)
                    print(root.data,end= " ")
                    self.inOrder(root.right)
            def postOrder(self, root:Node):
                if root is not None:
                    self.postOrder(root.left)
                    self.postOrder(root.right)
                    print(root.data,end= " ")

            def height(self, root:Node)->int:
                if root is None:
                    return 0
                leftHeight=self.height(root.left)
                rightHeight=self.height(root.right)
                if leftHeight>rightHeight:
                    return leftHeight+1
                return rightHeight+1

            def levelOrder(self, root):
                if root is None:
                    return
                temp=root
                que=[temp]
                while len(que)>0:
                    print(que[0].data,end= " ")
                    temp=que.pop(0)
                    if temp.left is not None:
                        que.append(temp.left)
                    if temp.right is not None:
                        que.append(temp.right)

            def levelWise(self, root:Node,level:int):
                if root is None:
                    return root
                if level==1:
                    print(root.data,end= " ")
                if level>1:
                    self.levelWise(root.left, level-1)
                    self.levelWise(root.right,level-1)

            def leftToRight(self,root:Node,level:int):
                if root is None:
                    return root
                if level==1:
                    print(root.data,end= " ")
                if level>1:
                    self.leftToRight(root.left, level-1)
                    self.leftToRight(root.right,level-1)
            def rightToLeft(self,root:Node,level:int):
                if root is None:
                    return root
                if level==1:
                    print(root.data,end= " ")
                if level>1:
                    self.rightToLeft(root.right,level-1)
                    self.rightToLeft(root.left, level-1)

            def zigZag(self, root:Node):
                flag=0
                h=self.height(root)
                for i in range(1,h+1):
                    if flag==0:
                        self.leftToRight(root,i)
                        print()
                        flag=1
                    else:
                        self.rightToLeft(root,i)
                        print()
                        flag=0
            def insert(self,root:Node,newNode:Node):
                if root is None:
                    root=newNode
                else:
                    if root.data>newNode.data:
                        if root.left is None:
                            root.left=newNode
                        else:
                            self.insert(root.left,newNode)
                    else:
                        if root.right is None:
                            root.right=newNode
                        else:
                            self.insert(root.right,newNode)
            def search(self,root:Node,val:int):
                if root==None or root.data=val:
                    return root
                if root.data>val:
                    return self.search(root.left,val)
                else:
                    return self.search(root.right,val)

        if __name__ == "__main__":
            root=Node(1)
            root.left=Node(2)
            root.right=Node(3)
            root.left.left=Node(4)
            root.left.right=Node(5)

            obj = Tree()
            obj.preOrder(root)
            print()
            obj.inOrder(root)
            print()
            obj.postOrder(root)
            print()
            print("H : ",obj.height(root))
            print()
            obj.levelOrder(root)
            print()
            obj.levelWise(root,2)
            print()
            obj.leftToRight(root,2)
            print()
            obj.rightToLeft(root,2)
            print()
            obj.zigZag(root)
            print("insert : ")
            obj=Tree()
            root=Node(1)
            obj.insert(5)
            obj.insert(2)
            obj.insert(6)
            obj.levelOrder(root)

40. Check subtree:

    Solutions:
        
        def areIdentical(root1, root2): 
            # Base Case 
            if root1 is None and root2 is None: 
                return True
            if root1 is None or root2 is None: 
                return False
            # Check fi the data of both roots is same and data of left and right subtrees are also same 
            return (root1.data == root2.data and areIdentical(root1.left , root2.left) and areIdentical(root1.right, root2.right) ) 
        
        # This function returns True if S is a subtree of T, otherwise False 
        def isSubtree(T, S): 
            # Base Case 
            if S is None: 
                return True
            if T is None: 
                return False
            # Check the tree with root as current node 
            if (areIdentical(T, S)): 
                return True
            # IF the tree with root as current node doesn't match  then try left and right subtreee one by one 
            return isSubtree(T.left, S) or isSubtree(T.right, S) 

41. Delete Node in BST:

        """
        When we delete a node, three possibilities arise.
        1) Node to be deleted is leaf: Simply remove from the tree.
                      50                            50
                   /     \         delete(20)      /   \
                  30      70       --------->    30     70 
                 /  \    /  \                     \    /  \ 
               20   40  60   80                   40  60   80
        2) Node to be deleted has only one child: Copy the child to the node and delete the child
                      50                            50
                   /     \         delete(30)      /   \
                  30      70       --------->    40     70 
                    \    /  \                          /  \ 
                    40  60   80                       60   80
        3) Node to be deleted has two children: Find inorder successor(min value in right subtree) of the node.  
        Copy contents of the inorder successor to the node and delete the inorder successor. Note that inorder predecessor can also be used.
                      50                            60
                   /     \         delete(50)      /   \
                  40      70       --------->    40    70 
                         /  \                            \ 
                        60   80                           80
        The important thing to note is, inorder successor is needed only when right child is not empty. 
        In this particular case, inorder successor can be obtained by finding the minimum value in right child of the node.
        """

42. Kth Largest Node in BST:
    [https://practice.geeksforgeeks.org/problems/kth-largest-element-in-bst/1/?track=amazon-bst&batchId=192#]
    
    Solutions:
    
        def kthLargest(self,root, k): #Time: O(h+k) space:O(h)
            def kthLargestUtil(root, k, c):
                nonlocal ans
                # Base cases, the second condition is important to avoid unnecessary recursive calls
                if root == None or c[0] >= k:
                    return
                # Follow reverse inorder traversal so that the largest element is visited first
                kthLargestUtil(root.right, k, c)
                # Increment count of visited nodes
                c[0] += 1
                # If c becomes k now, then this is the k'th largest
                if c[0] == k:
                    # print("K'th largest element is",root.data)
                    ans=root.data
                    return 
                # Recur for left subtree
                kthLargestUtil(root.left, k, c)
            ans=-1
            # Initialize count of nodes
            # visited as 0
            c = [0]
            kthLargestUtil(root, k, c)
            return ans
43. Inorder Successor of BST:
    [https://practice.geeksforgeeks.org/problems/inorder-successor-in-bst/1/?track=amazon-bst&batchId=192#]
    
    Solutions:
    
        Cases:
        
            1. If the right child of the node is not NULL then the inorder successor of this node will be the leftmost node in it’s right subtree.

            2. If the right child of node is NULL. Then we keep finding the parent of the given node x, say p such that p->left = x. 
        
            3. If the node is the rightmost node in the given tree. then there will be no inorder successor of this node NULL.
        
        # returns the inorder successor of the Node x in BST (rooted at 'root')
        def inorderSuccessor(self, root, x): #Time: O(n)
            def inorderSuccessorUtils(root, target_node):
                nonlocal next,ans
                # if root is None then return
                if(root == None):
                    return
                inorderSuccessorUtils(root.right, target_node)
                # if target node found, then enter this condition
                if(root.data == target_node.data):
                    # this will be true to the last node in inorder traversal i.e., rightmost node.
                    if(next == None):
                        # print ("inorder successor of",root.data , " is: None")
                        pass
                    else:
                        # print ( "inorder successor of",root.data , "is:", next.data)
                        ans=next
                next = root
                inorderSuccessorUtils(root.left, target_node)
            next = None
            ans=None
            inorderSuccessorUtils(root, x)
            return ans

44. Closest element in bst:
    [https://practice.geeksforgeeks.org/problems/find-the-closest-element-in-bst/1/?track=amazon-bst&batchId=192#]
    
    Solutions:
    
        def minDiff(self,root, k):  #time : O(h) space : O(h)
            # min_diff --> minimum difference till now 
            # min_diff_key --> node having minimum absolute  difference with K 
            def maxDiffUtil(ptr, k, min_diff, min_diff_key):
                if ptr == None: 
                    return
                # If k itself is present 
                if ptr.data == k:
                    min_diff_key[0] = k 
                    return
                # update min_diff and min_diff_key by checking current node value 
                if min_diff > abs(ptr.data - k):
                    min_diff = abs(ptr.data - k) 
                    min_diff_key[0] = ptr.data

                # if k is less than ptr->key then move in left subtree else in right subtree 
                if k < ptr.data:
                    maxDiffUtil(ptr.left, k, min_diff, min_diff_key)
                else:
                    maxDiffUtil(ptr.right, k, min_diff, min_diff_key)

            # Initialize minimum difference 
            min_diff, min_diff_key = 999999999999, [-1]

            # Find value of min_diff_key (Closest key in tree with k) 
            maxDiffUtil(root, k, min_diff, min_diff_key)

            return abs(k-min_diff_key[0]).  # min_diff_key[0] is the node data closes to k
    
45. kth smallest in bst:
    [https://practice.geeksforgeeks.org/problems/find-k-th-smallest-element-in-bst/1/?track=amazon-bst&batchId=192#]
    
    Solutions:
        
        class Solution:
            cnt=0
            ans=-1
            def inorder(self,root,k): #time : O(n)  space: O(1)
                if root is None:
                    return
                self.inorder(root.left,k)
                self.cnt+=1
                if k==self.cnt:
                    self.ans=root.data
                    return
                self.inorder(root.right,k)

            # Return the Kth smallest element in the given BST 
            def KthSmallestElement(self, root, k): 
                self.inorder(root,k)
                return self.ans

46. Pair with given sum in bst:
    [https://practice.geeksforgeeks.org/problems/find-a-pair-with-given-target-in-bst/1/?track=amazon-bst&batchId=192#]
    
    Solutions:
        
        class Solution:
            m=dict()
            x=-1
            def solve(self,root,target): 
                if root==None:
                    return
                self.solve(root.left,target)
                if target-root.data in self.m:
                    self.x=1
                else:
                    self.m[root.data]=1
                self.solve(root.right,target)

            def isPairPresent(self,root, target): #time: O(n) space:O(n)
                self.x=0
                self.m=dict()
                self.solve(root,target)
                return self.x

47. Heap Operations:
    
        """
        1. heappop(heap) - pop and return the smallest element from heap. i.e root
        2. heappush(heap,element) - push the value item onto the heap, maintaining heap invarient
        3. heapify(list) - transform list into heap, in place, in linear time
        4. heappushpop(heap, ele) - This function combines the functioning of both push and pop operations in one statement, increasing efficiency. Heap order is maintained after this operation.
        5. heapreplace(heap, ele) - This function also inserts and pops element in one statement, but it is different from above function. In this, element is first popped, then the element is pushed.i.e, the value larger than the pushed value can be returned. 
        """
        
        from heapq import heappush, heappop, heapify 
        class MinHeap:
            def __init__(self):
                self.heap = [] 

            def parent(self, i):
                return (i-1)/2.    # left child =2*i+1.  , right child=2*i+2     

            # Inserts a new key 'k'
            def insertKey(self, k):
                heappush(self.heap, k)           

            # Decrease value of key at index 'i' to new_val, It is assumed that new_val is smaller than heap[i]
            def decreaseKey(self, i, new_val):
                self.heap[i]  = new_val 
                while(i != 0 and self.heap[self.parent(i)] > self.heap[i]):
                    # Swap heap[i] with heap[parent(i)]
                    self.heap[i] , self.heap[self.parent(i)] = (
                    self.heap[self.parent(i)], self.heap[i])

            # Method to remove minium element from min heap
            def extractMin(self):
                return heappop(self.heap)

            # This functon deletes key at index i. It first reduces value to minus infinite and then calls extractMin()
            def deleteKey(self, i):
                self.decreaseKey(i, float("-inf"))
                self.extractMin()

            # Get the minimum element from the heap
            def getMin(self):
                return self.heap[0]

        # Driver pgoratm to test above function
        heapObj = MinHeap()
        heapObj.insertKey(3)
        heapObj.insertKey(2)
        heapObj.deleteKey(1)
        heapObj.insertKey(15)
        heapObj.insertKey(5)
        heapObj.insertKey(4)
        heapObj.insertKey(45)
        print heapObj.extractMin(),
        print heapObj.getMin(),
        heapObj.decreaseKey(2, 1)
        print heapObj.getMin()
        
47.2. Heap implementation priority queue implementation:

    Solution:
        
        # Priority Queue implementation in Python
        def heapify(arr, n, i):
            # Find the largest among root, left child and right child
            largest = i
            l = 2 * i + 1
            r = 2 * i + 2
            if l < n and arr[i] < arr[l]:
                largest = l
            if r < n and arr[largest] < arr[r]:
                largest = r
            # Swap and continue heapifying if root is not largest
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)

        # Function to insert an element into the tree
        def insert(array, newNum):
            size = len(array)
            if size == 0:
                array.append(newNum)
            else:
                array.append(newNum)
                for i in range((size // 2) - 1, -1, -1):
                    heapify(array, size, i)

        # Function to delete an element from the tree
        def deleteNode(array, num):
            size = len(array)
            i = 0
            for i in range(0, size):
                if num == array[i]:
                    break
            array[i], array[size - 1] = array[size - 1], array[i]
            array.remove(size - 1)
            for i in range((len(array) // 2) - 1, -1, -1):
                heapify(array, len(array), i)
        arr = []
        insert(arr, 3)
        insert(arr, 4)
        insert(arr, 9)
        insert(arr, 5)
        insert(arr, 2)
        print ("Max-Heap array: " + str(arr))
        deleteNode(arr, 4)
        print("After deleting an element: " + str(arr))


48. Kth smallest element:
    [https://practice.geeksforgeeks.org/problems/kth-element-in-matrix/1/?track=amazon-heap&batchId=192#]
    
    Solutions:
        
        import heapq
        def kthSmallest(mat, n, k):  #time : O(klog(n)  space: O(n)
            arr=[]
            for i in mat:
                arr+=i
            brr=heapq.nlargest(len(arr)-k+1,arr) # or return heapq.nsmallest(len(arr)-k+1,arr)
            return brr[-1]
           
49. Min cost of ropes:
    [https://practice.geeksforgeeks.org/problems/minimum-cost-of-ropes-1587115620/1/?track=amazon-heap&batchId=192#]
    
        """
        There are given N ropes of different lengths, we need to connect these ropes into one rope. The cost to connect two ropes is equal to sum of their lengths. The task is to connect the ropes with minimum cost.

        Example:

            Input:
            n = 4
            arr[] = {4, 3, 2, 6}
            Output: 
            29
            Explanation:
            For example if we are given 4 ropes of lengths 4, 3, 2 and 6. We can connect the ropes in following ways.
            1) First connect ropes of lengths 2 and 3. Now we have three ropes of lengths 4, 6 and 5.
            2) Now connect ropes of lengths 4 and 5. Now we have two ropes of lengths 6 and 9.
            3) Finally connect the two ropes and all ropes have connected.
            Total cost for connecting all ropes is 5 + 9 + 15 = 29. This is the optimized cost for connecting ropes.
        """
    
    Solutions:
        
        import heapq
        class Solution:
            #Function to return the minimum cost of connecting the ropes.
            cost=0
            def minCost(self,arr,n) : #Time: O(nlogn). space: O(n)
                heapq.heapify(arr)
                while len(arr)>1:
                    r1=heapq.heappop(arr)
                    r2=heapq.heappop(arr)
                    # print("popped : ",r1,r2,arr)
                    self.cost+=r1+r2
                    heapq.heappush(arr,r1+r2)
                    # print("pushed : ",r1+r2,arr)
                return self.cost

50. Find median in the stream:
    [https://practice.geeksforgeeks.org/problems/find-median-in-a-stream-1587115620/1/?track=amazon-heap&batchId=192#]
    
        """
        Given an input stream of N integers. The task is to insert these numbers into a new stream and find the median of the stream formed by each insertion of X to the new stream.
        *median of odd n= n/2,   median of even n = avg(n/2,n/2+1)
        Example:
        Input:
        N = 4
        X[] = 5,15,1,3
        Output:
        5
        10
        5
        4
        Explanation:Flow in stream : 5, 15, 1, 3 
        5 goes to stream --> median 5 (5) 
        15 goes to stream --> median 10 (5,15) 
        1 goes to stream --> median 5 (5,15,1) 
        3 goes to stream --> median 4 (5,15,1 3)
        """
        
    Solutions:
        
        import heapq
        class Solution:  #time: O(nlogn + logn).  space: O(n)
            arr=[]
            def balanceHeaps(self):
                #Balance the two heaps size , such that difference is not more than one.
                heapq.heapify(self.arr)
                
            def getMedian(self)->float:
                # return the median of the data received till now.
                self.arr.sort()
                n = len(self.arr)
                if not self.arr or n == 1:
                    return 0 or self.arr[0]            # 0 or x= x if x!=0 else 1
                elif n & 1:     # odd sized array
                    return self.arr[n // 2]
                else:      #even sized array
                    return (self.arr[n // 2] + self.arr[(n // 2) - 1]) / 2
                    
            def insertHeaps(self,x)->None:
                heapq.heappush(self.arr,x)
                self.balanceHeaps()
                
    Solutions:
    
        """
        1. If the current element to be added is less than the maximum element of the max heap, then add this to the max heap.
        2. If the difference between the size of the max and min heap becomes greater than 1, the top element of the max heap is removed and added to the min-heap.
        3. If the current element to be added is greater than the maximum element of the min-heap, then add this to the min-heap.
        4. If the difference between the size of the max and min heap becomes greater than 1, the top element of the min-heap is removed and added to the max heap.
        """
        
        import heapq
        class Solution:  #Time : O(nlogn). space: O(n)
            small = []   # max heap to store smaller half
            large = []   # min heap to store larger half
            
            def getMedian(self):
                # return the median of the data received till now.
                if len(self.small) == len(self.large):
                    return (-self.small[0]+self.large[0])/2
                elif len(self.small) > len(self.large):
                    return -self.small[0]
                return self.large[0]

            def insertHeaps(self,x):
                #:param x: value to be inserted
                if not self.small or x <= -self.small[0]:
                    heapq.heappush(self.small, -x)
                else:
                    heapq.heappush(self.large, x)

                while len(self.small) - len(self.large) > 1:
                    val = -heapq.heappop(self.small)
                    heapq.heappush(self.large, val)

                while len(self.large) - len(self.small) > 1:
                    val = -heapq.heappop(self.large)
                    heapq.heappush(self.small, val)

51. Rearrange characters:
    [https://practice.geeksforgeeks.org/problems/rearrange-characters5322/1/?track=amazon-heap&batchId=192#]
    
        """
        Given a string S such that it may contain repeated lowercase alphabets. Rearrange the characters in the string such that no two adjacent characters are same.
        Example :
        Input:
        S = geeksforgeeks
        Output: 1
        Explanation: egeskerskegof can be one way of
        rearranging the letters.
        """
    
    Solutions:
    
        def getMaxCountChar(count):
          maxCount = 0
          for i in range(26):
            if count[i] > maxCount:
                maxCount = count[i]
                maxChar = chr(i + ord('a'))
          return maxCount, maxChar

        #Function to rearrange the characters in a string such that no two adjacent characters are same.
        def rearrangeString(S):  #Time : O(nlogn). space: O(1)
            n = len(S)
            
            # if length of string is None return False
            if not n:
              return ""  #False
              
            # create a hashmap for the alphabets
            count = [0] * 26
            for char in S:
              count[ord(char) - ord('a')] += 1
            maxCount, maxChar = getMaxCountChar(count)
            
            # if the char with maximum frequency is more than the half of the total length of the string than return False
            if maxCount > (n + 1) // 2:
              return ""  #False
              
            # create a list for storing the result
            res = [None] * n
            ind = 0
            
            # place all occurrences of the char with maximum frequency in even positions
            while maxCount:
              res[ind] = maxChar
              ind += 2
              maxCount -= 1
              
            # replace the count of the char with maximum frequency to zero as all the maxChar are already placed in the result
            count[ord(maxChar) - ord('a')] = 0
            
            # place all other char in the result starting from remaining even positions and then place in the odd positions
            for i in range(26):
              while count[i] > 0:
                  if ind >= n:
                      ind = 1
                  res[ind] = chr(i + ord('a') )
                  ind += 2
                  count[i] -= 1
                  
            if len(res)==0:
                return ""  #False
            s=""
            for i in res:
                s+=i
            return s

52. Nearly Sorted array:
    [https://practice.geeksforgeeks.org/problems/nearly-sorted-1587115620/1/?track=amazon-heap&batchId=192#]
    
        """
        Given an array of n elements, where each element is at most k away from its target position, you need to sort the array optimally.
        Example :
        Input:
        n = 7, k = 3
        arr[] = {6,5,3,2,8,10,9}
        Output: 2 3 5 6 8 9 10
        Explanation: The sorted array will be
        2 3 5 6 8 9 10
        """
    
    Solutions:
        
        import heapq
        class Solution: #Time : O(k) + O((m) * log(k)) where m = n – k and  space: O(k) => O(nlogk) time, O(n) space

            #Function to return the sorted array.
            def nearlySorted(self,arr,n,k):
                # return heapq.nsmallest(n,a)[::-1]  # this is also same

                # List of first k+1 items
                heap = arr[:k + 1]

                # using heapify to convert list into heap(or min heap)
                heapq.heapify(heap)

                # "rem_elmnts_index" is index for remaining elements in arr and "target_index" is target index of for current minimum element in Min Heap "heap".
                target_index = 0
                for rem_elmnts_index in range(k + 1, n):
                    arr[target_index] = heapq.heappop(heap)
                    heapq.heappush(heap, arr[rem_elmnts_index])
                    target_index += 1

                while heap:
                    arr[target_index] = heapq.heappop(heap)
                    target_index += 1
                return arr

53. BFS graph:
    [https://practice.geeksforgeeks.org/problems/bfs-traversal-of-graph/1/?track=amazon-graphs&batchId=192#]  
    
    Solutions:
        
        # This class represents a directed graphvusing adjacency list representation
        from collections import defaultdict
        class Solution:
            def __init__(self):
                self.graph = defaultdict(list)
            def addEdge(self,u,v):
                self.graph[u].append(v)

            def BFS(self, s,vertex,traverse):  #time : O(V + E). space: O(V)
                # Mark all the vertices as not visited
                visited = [False] *( (vertex) + 1)
                # Create a queue for BFS
                queue = []

                # Mark the source node as visited and enqueue it
                queue.append(s)
                visited[s] = True

                while queue:
                    # Dequeue a vertex from queue and print it
                    s = queue.pop(0)
                    # print (s, end = " ")
                    traverse.append(s)

                    # Get all adjacent vertices of the dequeued vertex s. If a adjacent has not been visited, then mark it visited and enqueue it
                    for i in self.graph[s]:
                        if visited[i] == False:
                            queue.append(i)
                            visited[i] = True
                return traverse			
            def bfsOfGraph(self, V, adj):
                vertex=V
                traverse=[]
                for i in range(len(adj)):
                    for j in range(len(adj[i])):
                        if adj[i][j]!=[]:
                            self.addEdge(i,adj[i][j])
                return self.BFS(0,vertex,traverse)

54. DFS Graph:
    [https://practice.geeksforgeeks.org/problems/depth-first-traversal-for-a-graph/1/?track=amazon-graphs&batchId=192#]
    
    Solutions:
        
        from collections import defaultdict
        class Solution:
            def __init__(self):
                self.graph = defaultdict(list)
            def addEdge(self,u,v):
                self.graph[u].append(v)
            # A function used by DFS
            def DFSUtil(self, v, visited,traverse):
                # Mark the current node as visited and print it
                visited.add(v)
                # print(v, end=' ')
                traverse.append(v)
                # Recur for all the vertices adjacent to this vertex
                for neighbour in self.graph[v]:
                    if neighbour not in visited:
                        self.DFSUtil(neighbour, visited,traverse)
                return traverse
            # The function to do DFS traversal. It uses recursive DFSUtil()
            def DFS(self, v):
                # Create a set to store visited vertices
                visited = set()
                traverse=[]
                # Call the recursive helper function to print DFS traversal
                return self.DFSUtil(v, visited,traverse)

            #Function to return a list containing the DFS traversal of the graph.
            def dfsOfGraph(self, V, adj):
                vertex=V
                for i in range(len(adj)):
                    for j in range(len(adj[i])):
                        if adj[i][j]!=[]:
                            self.addEdge(i,adj[i][j])
                return self.DFS(0)

55. Rotten Oranges:
    [https://practice.geeksforgeeks.org/problems/rotten-oranges2536/1/?track=amazon-graphs&batchId=192#]
    
        """
        Given a grid of dimension nxm where each cell in the grid can have values 0, 1 or 2 which has the following meaning:
        a) 0 : Empty cell
        b) 1 : Cells have fresh oranges
        c) 2 : Cells have rotten oranges
        We have to determine what is the minimum time required to rot all oranges. 
        A rotten orange at index [i,j] can rot other fresh orange at indexes [i-1,j], [i+1,j], [i,j-1], [i,j+1] (up, down, left and right) in unit time. 

        Example :
        Input: grid = {{0,1,2},{0,1,2},{2,1,1}}
        Output: 1
        Explanation: The grid is-
        0 1 2
        0 1 2
        2 1 1
        Oranges at positions (0,2), (1,2), (2,0)
        will rot oranges at (0,1), (1,1), (2,2) and 
        (2,1) in unit time.
        """
    
    Solutions:
    
        #Function to find minimum time required to rot all oranges. 
        def orangesRotting(self, grid):  #Time: O(n*m).  space: O(n)
            rots = set() 
            fresh = set()
            for i in range(len(grid)): 
                for j in range(len(grid[i])): 
                    if grid[i][j] == 1: 
                        fresh.add((i,j))
                    if grid[i][j] == 2: 
                        rots.add((i,j))
            if len(fresh) == 0: 
                return 0
            if len(rots) == 0: 
                return -1
            depth = 0
            q = []
            vis = set()
            for cord in rots: 
                q.append(cord)
                vis.add(cord)

            # print(len(q))
            while q: 
                q_len = len(q)
                for k in range(q_len): 
                    i=q[k][0]
                    j=q[k][1]
                    vis.add(q[k])
                    if i > 0 and grid[i-1][j] == 1 and (i-1,j) not in vis:
                        q.append((i-1,j))
                        vis.add((i-1,j))
                        fresh.discard((i-1,j))
                    if i < len(grid)-1 and grid[i+1][j] == 1 and (i+1,j) not in vis:
                        q.append((i+1,j))
                        vis.add((i+1,j))
                        fresh.discard((i+1,j))
                    if j > 0 and grid[i][j-1] == 1 and (i,j-1) not in vis:
                        q.append((i,j-1))
                        vis.add((i,j-1))
                        fresh.discard((i,j-1))
                    if j < len(grid[i])-1 and grid[i][j+1] == 1 and (i,j+1) not in vis:
                        q.append((i,j+1))
                        vis.add((i,j+1))
                        fresh.discard((i,j+1))

                q=q[q_len:]
                if q: 
                    depth+=1
            if len(fresh) > 0: 
                return -1
            return depth

56. Detect Cycle in directed Graph:
    [https://practice.geeksforgeeks.org/problems/detect-cycle-in-a-directed-graph/1/?track=amazon-graphs&batchId=192#]
    
    Solutions:
    
        from collections import defaultdict
        class Solution:  #time: O(v+e). space O(v)
            def __init__(self):
                self.graph = defaultdict(list)
            def addEdge(self,u,v):
                self.graph[u].append(v)
            def isCyclicUtil(self, v, visited, recStack):
                # Mark current node as visited and adds to recursion stack
                visited[v] = True
                recStack[v] = True

                # Recur for all neighbours if any neighbour is visited and in recStack then graph is cyclic
                for neighbour in self.graph[v]:
                    if visited[neighbour] == False:
                        if self.isCyclicUtil(neighbour, visited, recStack) == True:
                            return True
                    elif recStack[neighbour] == True:
                        return True

                # The node needs to be poped from recursion stack before function ends
                recStack[v] = False
                return False
            def isCyclic(self, V, adj):
                for i in range(len(adj)):
                    for j in range(len(adj[i])):
                        if adj[i][j]!=[]:
                            self.addEdge(i,adj[i][j])
                visited = [False] * (V + 1)
                recStack = [False] * (V + 1)
                for node in range(V):
                    if visited[node] == False:
                        if self.isCyclicUtil(node,visited,recStack) == True:
                            return True
                return False

57. Detect cycle in undirected graph:
    [https://practice.geeksforgeeks.org/problems/detect-cycle-in-an-undirected-graph/1/?track=amazon-graphs&batchId=192#]
    
    Solutions:
    
        class Solution:   #time: O(v+e). space O(v)
            def isCycle(self, V, adj):
                visited = [False] * V
                for i in range(V):
                    if visited[i] == False:
                        if self.cycledfs(i,visited,-1,adj):
                            return True
                return False

            def cycledfs(self,src,visited,parent,adj):
                visited[src] = True
                for neigh in adj[src]:
                    if visited[neigh] == False:
                        if self.cycledfs(neigh,visited,src,adj):
                            return True
                    elif neigh != parent:
                        return True
                return False
            
58. Find whether path exists:
    [https://practice.geeksforgeeks.org/problems/find-whether-path-exist5238/1/?track=amazon-graphs&batchId=192#]
    
        """
        Given a grid of size n*n filled with 0, 1, 2, 3. Check whether there is a path possible from the source to destination. 
        You can traverse up, down, right and left.
        The description of cells is as follows:

        A value of cell 1 means Source.
        A value of cell 2 means Destination.
        A value of cell 3 means Blank cell.
        A value of cell 0 means Wall.
        
        Input: grid = {{3,0,3,0,0},{3,0,0,0,3},{3,3,3,3,3},{0,2,3,0,0},{3,0,0,1,3}}
        Output: 0
        Explanation: The grid is-
        3 0 3 0 0 
        3 0 0 0 3 
        3 3 3 3 3 
        0 2 3 0 0 
        3 0 0 1 3 
        There is no path to reach at (3,1) i,e at destination from (4,3) i,e source.
        """

    Solutions:
        
        class Solution:  #Time: O(n2).  space: O(n2)
            def helper_dfs(self,i,j,visited,grid):
                if i>=0 and i<len(grid) and j>=0 and j<len(grid[0]) and grid[i][j]!=0 and visited[i][j] ==False:
                    visited[i][j]=True
                    if grid[i][j]==2:
                        return True
                    d=self.helper_dfs(i+1,j,visited,grid)
                    u=self.helper_dfs(i-1,j,visited,grid)
                    r=self.helper_dfs(i,j+1,visited,grid)
                    l=self.helper_dfs(i,j-1,visited,grid)
                    if d==True or u==True or r==True or  l==True:
                        return True
                    else:
                        return False
            #Function to find whether a path exists from the source to destination.
            def is_Possible(self, grid):
                a=len(grid)
                b=len(grid[0])
                visited=[[False for i in range(b)]for i in range(a)]
                for i in range(0,a):
                    for j in range(0,b):
                        if grid[i][j]==1:
                            c=self.helper_dfs(i,j,visited,grid)
                            break
                if c==True:
                    return 1
                else:
                    return 0
                
59. Count possible paths between two vertices:
    [https://practice.geeksforgeeks.org/problems/possible-paths-between-2-vertices-1587115620/1/?track=amazon-graphs&batchId=192#]
    
    Solutions:
    
        class Solution:  #Time :O(V).  Space: O(V)
            visited=[]
            def dfs(self,s,d,adj):
                self.visited
                if s==d['end']:
                    d['count']+=1
                    return
                self.visited[s]=1
                for i in range(len(adj[s])):
                    if self.visited[adj[s][i]]==0:
                        self.dfs(adj[s][i],d,adj)
                self.visited[s]=0

            #Function to count paths between two vertices in a directed graph.
            def countPaths(self, V, adj, source, destination):
                self.visited
                # print(adj)
                self.visited=[0]*V
                d={'end':destination,'count':0}
                self.dfs(source,d,adj)
                return d['count']

60. Find number of islands:
    [https://practice.geeksforgeeks.org/problems/find-the-number-of-islands/1/?track=amazon-graphs&batchId=192#]
    
        """
        Given a grid of size n*m (n is number of rows and m is number of columns grid has) consisting of '0's(Water) and '1's(Land). 
        Find the number of islands.
        Note: An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically or diagonally 
        i.e., in all 8 directions.
        
        Input:
        grid = {{0,1},{1,0},{1,1},{1,0}}
        Output:
        1
        Explanation:
        The grid is-
        0 1
        1 0
        1 1
        1 0
        All lands are connected.
        """
    
    Solutions:
    
        class Solution: #Time: O(n*m).  space: O(n*m)
            def numIslands(self,grid):
                def bfs(r,c):
                    queue = []
                    visitSet.add((r,c))
                    queue.append((r,c))

                    while queue:
                        row,col = queue.pop(0)
                        directions = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[-1,1],[1,-1]]
                        for dr,dc in directions:
                            r,c = row+dr,col+dc
                            if (r in range(ROWS) and 
                                c in range(COLS) and 
                                grid[r][c] == 1 and 
                                (r,c) not in visitSet):
                                    visitSet.add((r,c))
                                    queue.append((r,c))

                visitSet = set()
                isLands = 0
                ROWS,COLS = len(grid),len(grid[0])
                for i in range(ROWS):
                    for j in range(COLS):
                        if (grid[i][j] == 1 and (i,j) not in visitSet):
                            bfs(i,j)
                            isLands += 1
                return isLands

61. Unit area of largest region of 1's.
    [https://practice.geeksforgeeks.org/problems/length-of-largest-region-of-1s-1587115620/1/?track=amazon-graphs&batchId=192#]
    
    Solutions:
        
        class Solution:  #Time :O(n*m). space: O(n*m)
            ROW=0
            COL=0
            def isSafe(self,M, row, col, visited):
                return ((row >= 0) and (row < self.ROW) and (col >= 0) and 
                (col < self.COL) and (M[row][col] and not visited[row][col]))

            def DFS(self,M, row, col, visited, count):
                # These arrays are used to get row and column numbers of 8 neighbours of a given cell
                rowNbr = [-1, -1, -1, 0, 0, 1, 1, 1]
                colNbr = [-1, 0, 1, -1, 1, -1, 0, 1]

                # Mark this cell as visited
                visited[row][col] = True

                # Recur for all connected neighbours
                for k in range(8):
                    if (self.isSafe(M, row + rowNbr[k],col + colNbr[k], visited)):

                        # increment region length by one
                        count[0] += 1
                        self.DFS(M, row + rowNbr[k],col + colNbr[k], visited, count)

            #Function to find unit area of the largest region of 1s.
            def findMaxArea(self, grid):
                self.ROW=len(grid)
                self.COL=len(grid[0])
                # Make a bool array to mark visited cells. Initially all cells are unvisited
                visited = [[0] * self.COL for i in range(self.ROW)]

                # Initialize result as 0 and traverse through the all cells of given matrix
                result = -999999999999
                for i in range(self.ROW):
                    for j in range(self.COL):
                        # If a cell with value 1 is not
                        if (grid[i][j] and not visited[i][j]):
                            # visited yet, then new region found
                            count = [1]
                            self.DFS(grid, i, j, visited, count)
                            # maximum region
                            result = max(result, count[0])
                return result

62. Topological Sort:
    [https://practice.geeksforgeeks.org/problems/topological-sort/1/?track=amazon-graphs&batchId=192#]
    
        """
        Topological sorting for Directed Acyclic Graph (DAG) is a linear ordering of vertices such that for every directed edge u v,
        vertex u comes before v in the ordering. Topological Sorting for a graph is not possible if the graph is not a DAG.
        """
        
    Solutions:
        
        from collections import defaultdict
        class Solution:  #Time : O(V+E). space : O(V)
            def __init__(self):
                self.graph = defaultdict(list)
            def addEdge(self,u,v):
                self.graph[u].append(v)
            def topologicalSortUtil(self, v, visited, stack):
                # Mark the current node as visited.
                visited[v] = True

                # Recur for all the vertices adjacent to this vertex
                for i in self.graph[v]:
                    if visited[i] == False:
                        self.topologicalSortUtil(i, visited, stack)

                # Push current vertex to stack which stores result
                stack.append(v)

            #Function to return list containing vertices in Topological order.
            def topoSort(self, V, adj):
                vertex=V
                # print(adj)
                for i in range(len(adj)):
                    for j in range(len(adj[i])):
                        if adj[i][j]!=[]:
                            self.addEdge(i,adj[i][j])

                # Mark all the vertices as not visited
                visited = [False]*V
                stack = []

                # Call the recursive helper function to store Topological
                # Sort starting from all vertices one by one
                for i in range(V):
                    if visited[i] == False:
                        self.topologicalSortUtil(i, visited, stack)

                # Print contents of the stack
                return stack[::-1]  # return list in reverse order

63. Word Search:
    [https://practice.geeksforgeeks.org/problems/word-search/1/?track=amazon-graphs&batchId=192#]
    
        """
        Given a 2D board of letters and a word. Check if the word exists in the board. 
        The word can be constructed from letters of adjacent cells only. ie - horizontal or vertical neighbors.
        The same letter cell can not be used more than once.
        Example :
        Input: board = {{a,g,b,c},{q,e,e,l},{g,b,k,s}},
        word = "geeks"
        Output: 1
        Explanation: The board is-
        a *g b c
        q *e *e l
        g b *k *s
        The letters which are used to make the "geeks" are marked *.
        """
        
    Solutions:
    
        class Solution:  #time : O(N * M * 4L)  space: O(L) ; L = Length of word
            def findPath(self,index, i, j, usedIndexs):
                if i < 0 or j < 0 or i >= len(board) or j >= len(board[0]):
                    return False

                if word[index] != board[i][j]:
                    return False

                currIndex = str(i) + str(j)
                if currIndex in usedIndexs: 
                    return False

                usedIndexs.add(currIndex)

                if len(word) - 1 == index:
                    return True

                return self.findPath(index + 1, i + 1, j, usedIndexs) or self.findPath(index + 1, i - 1, j, usedIndexs) 
                or self.findPath(index + 1, i, j + 1, usedIndexs) or self.findPath(index + 1, i, j - 1, usedIndexs)

            def isWordExist(self, board, word):
                letterIndexs = []
                for i in range(len(board)):
                    for j in range(len(board[0])):
                        if board[i][j] == word[0]:
                            letterIndexs.append({ 'i': i, 'j': j })
                if len(letterIndexs) == 0:
                    return 0
                output = False

                for indexs in letterIndexs:
                    usedIndexs = set()
                    output = self.findPath(0, indexs['i'], indexs['j'], usedIndexs)
                    if output:
                        return 1
                return 0

64. Graph
    
    Solutions:
        
        # This class represents a directed graphvusing adjacency list representation
        from collections import defaultdict
        class Solution:
            def __init__(self):
                self.graph = defaultdict(list)
            def addEdge(self,u,v):
                self.graph[u].append(v)
            def getVertices(self):
        return list(self.gdict.keys())

            def edges(self):
                return self.findedges()

            def findedges(self):
                edgename=[]
                for vertx in self.graph:
                    for nxtvertx in self.graph[vertx]:
                        if {vertx,nxtvertx} not in edgename:
                            edgename.append({vertx,nxtvertx})
                return edgename

            def function(self, V, adj):
                vertex=V
                for i in range(len(adj)):
                    for j in range(len(adj[i])):
                        if adj[i][j]!=[]:
                            self.addEdge(i,adj[i][j])
                
65. Minimum Spanning tree:
    [https://practice.geeksforgeeks.org/problems/minimum-spanning-tree/1/?track=amazon-graphs&batchId=192#]
    
        """
        Given a weighted, undirected and connected graph of V vertices and E edges. The task is to find the sum of weights of the edges of the Minimum Spanning Tree.
        """
        
        """
        a) Given a connected and undirected graph, a spanning tree of that graph is a subgraph that is a tree and connects all the vertices together. A single graph can have many different spanning trees. 
        b) A minimum spanning tree has (V – 1) edges where V is the number of vertices in the given graph. 
        
        c) finding MST using Kruskal’s algorithm

            1. Sort all the edges in non-decreasing order of their weight. 
            2. Pick the smallest edge. Check if it forms a cycle with the spanning tree formed so far. 
               If cycle is not formed, include this edge. Else, discard it. 
            3. Repeat step#2 until there are (V-1) edges in the spanning tree.

        """
        
    Solutions:
        
        import heapq
        class Solution: #Time: O(ElogV).   space: O(V2)
            #Function to find sum of weights of edges of the Minimum Spanning Tree.
            def spanningTree(self, V, adj):
                key=[float("inf")]*V
                key[0]=0
                mst = [False]*V
                parent = [-1]*V
                pq=[]   
                heapq.heappush(pq, (key[0],0))       #key[i],index

                while pq:
                    node, ind = heapq.heappop(pq)
                    mst[ind]=True

                    for j in adj[ind]:
                        k=j[0]
                        wt=j[1]
                        if(mst[k]==False and wt < key[k]):
                            parent[k]=ind
                            key[k]=wt
                            heapq.heappush(pq, (key[k],k))
                return sum(key)
       
     Solutions:
        
        import heapq
        class Solution: #Time: O(ElogV).   space: O(V2)
            #Function to find sum of weights of edges of the Minimum Spanning Tree.
            def spanningTree(self, V, adj):
                 key=[float('inf')]*V
                 visited=[False]*V
                 key[0]=0
                 for c in range(V-1):
                     mini=float('inf')
                     for v in range(V):
                         if visited[v]==False and key[v]<mini:
                             mini=key[v]
                             u=v
                     visited[u]=True
                     for v,wt in adj[u]:
                         if visited[v]==False and wt<key[v]:
                             key[v]=wt
                 return sum(key)
        
    Solutions:
       
        class Solution: #Time: O(ElogV).   space: O(V2)
            def __init__(self):
                self.graph = []
            def addEdge(self, u, v, w):
                self.graph.append([u, v, w])

            # A utility function to find set of an element i
            def find(self, parent, i):
                if parent[i] == i:
                    return i
                return self.find(parent, parent[i])

            # A function that does union of two sets of x and y based on rank
            def union(self, parent, rank, x, y):
                xroot = self.find(parent, x)
                yroot = self.find(parent, y)

                # Attach smaller rank tree under root of high rank tree (Union by Rank)
                if rank[xroot] < rank[yroot]:
                    parent[xroot] = yroot
                elif rank[xroot] > rank[yroot]:
                    parent[yroot] = xroot
                # If ranks are same, then make one as root and increment its rank by one
                else:
                    parent[yroot] = xroot
                    rank[xroot] += 1

            # The main function to construct MST using Kruskal's algorithm
            def KruskalMST(self,V):
                result = []  # This will store the resultant MST
                # An index variable, used for sorted edges
                i = 0
                # An index variable, used for result[]
                e = 0

                # Step 1:  Sort all the edges in non-decreasing order of their weight.  If we are not allowed to change the given graph, we can create a copy of graph
                self.graph = sorted(self.graph,key=lambda item: item[2])
                parent = []
                rank = []
                # Create V subsets with single elements
                for node in range(V):
                    parent.append(node)
                    rank.append(0)

                # Number of edges to be taken is equal to V-1
                while e < V - 1:
                    # Step 2: Pick the smallest edge and increment the index for next iteration
                    u, v, w = self.graph[i]
                    i = i + 1
                    x = self.find(parent, u)
                    y = self.find(parent, v)

                    # If including this edge does't cause cycle, include it in result and increment the indexof result for next edge
                    # Else discard the edge
                    if x != y:
                        e = e + 1
                        result.append([u, v, w])
                        self.union(parent, rank, x, y)
                    
                minimumCost = 0
                # print ("Edges in the constructed MST")
                for u, v, weight in result:
                    minimumCost += weight
                    # print("%d -- %d == %d" % (u, v, weight))
                # print("Minimum Spanning Tree" , minimumCost)
                return minimumCost

            #Function to find sum of weights of edges of the Minimum Spanning Tree.
            def spanningTree(self, V, adj):
                # print(adj)
                for i in range(len(adj)):
                    for j in range(len(adj[i])):
                        if adj[i][j]!=[]:
                            self.addEdge(i,adj[i][j][0],adj[i][j][1])
                return self.KruskalMST(V)

66. Dijkstra  Algo for Minimum spanning tree in graph:
    [https://practice.geeksforgeeks.org/problems/implementing-dijkstra-set-1-adjacency-matrix/1/?track=amazon-graphs&batchId=192#]
    
        """
        Given a weighted, undirected and connected graph of V vertices and E edges, 
        Find the shortest distance of all the vertex's from the source vertex S.
        """
        
    Solutions:
    
        from collections import deque
        class Solution:  #Time : O(V2).  space: O(V2)
            #Function to find the shortest distance of all the vertices from the source vertex S.
            def dijkstra(self, V, adj, S):
                dist = [float("inf")]*V
                dist[S] = 0
                q = deque()
                q.append(S)
                while q:
                    node = q.pop()
                    for new_node,next_dist in adj[node]:
                        if dist[node] + next_dist < dist[new_node]:
                            dist[new_node] = dist[node] + next_dist
                            q.append(new_node)
                return dist

67. Minimum Cost Path in grid:
    [https://practice.geeksforgeeks.org/problems/minimum-cost-path3833/1/?track=amazon-graphs&batchId=192#]
      
        """
        Given a square grid of size N, each cell of which contains integer cost which represents a cost to traverse through that cell, 
        we need to find a path from top left cell to bottom right cell by which the total cost incurred is minimum.
        From the cell (i,j) we can go (i,j-1), (i, j+1), (i-1, j), (i+1, j). 
        
        Example 1:
        Input: grid = {{9,4,9,9},{6,7,6,4},
        {8,3,3,7},{7,4,9,10}}
        Output: 43
        Explanation: The grid is-
        9 4 9 9
        6 7 6 4
        8 3 3 7
        7 4 9 10
        The minimum cost is- 9 + 4 + 7 + 3 + 3 + 7 + 10 = 43.
        """
        
    Solutions:
        
        import heapq
        class Solution:  #time : O(n2*log(n)).  space:O(n2)

            #Function to return the minimum cost to react at bottom right cell from top left cell.
            # Using Dijkstra's Algo  using Heap  for 2D array
            def minimumCostPath(self, grid):
                m=len(grid)
                n=len(grid[0])
                i=0 
                j=0
                dist=[[float("inf")]*n for i in range(m)]
                dist[i][j]=grid[i][j]
                pq=[]
                dx = [0, 0, 1, -1]
                dy = [1, -1, 0, 0]
                heapq.heappush(pq,(grid[0][0],(0,0)))
                while pq:
                    wt,nei=heapq.heappop(pq)
                    u=nei[0]
                    v=nei[1]
                    for k in range(4):
                        nu=u+dx[k]
                        nv=v+dy[k]
                        if 0<=nu<n and 0<=nv<m and wt+grid[nu][nv]<dist[nu][nv]:
                            nwt=wt+grid[nu][nv]
                            dist[nu][nv]=nwt
                            heapq.heappush(pq,(nwt,(nu,nv)))
                return dist[n-1][m-1]

68. Bridge edge in a graph:
    [https://practice.geeksforgeeks.org/problems/bridge-edge-in-graph/1/?track=amazon-graphs&batchId=192#]
    
    """
    Given a Graph of V vertices and E edges and another edge(c - d), the task is to find if the given edge is a Bridge. 
    i.e., removing the edge disconnects the graph.
    """
    
    Solutions:
    
        def isBridge(self, V, adj, c, d):  #Time:  O(V + E).   space: O(V)
            def dfs(node,adj,vis):
                vis[node]=1
                for j in adj[node]:
                    if vis[j]==0:
                        dfs(j,adj,vis)
            vis1=[0]*V   
            initial=0
            for i in range(V):
                if vis1[i]==0:
                    initial+=1
                    dfs(i,adj,vis1)

            vis2=[0]*V
            if d in adj[c]:
                adj[c].remove(d)
            if c in adj[d]:
                adj[d].remove(c)
            final=0
            for i in range(V):
                if vis2[i]==0:
                    final+=1
                    dfs(i,adj,vis2)
            if initial==final:
                return 0
            else:
                return 1
                
69. Soduku Solve:
    [https://practice.geeksforgeeks.org/problems/solve-the-sudoku-1587115621/1/?track=Amazon-backtracking&batchId=192#]
    
    Solutions:

        class Solution: #time: O(9^(n*n)). space O(n2)
            def is_valid(self, grid, i, j, k):
                n = len(grid)

                for p in range(n):
                    if grid[i][p] == k:
                        return False
                    if grid[p][j] == k:
                        return False

                sq_top_left = [(i//3)*3, (j//3)*3]
                for p in range(sq_top_left[0], sq_top_left[0]+3):
                    for q in range(sq_top_left[1], sq_top_left[1]+3):
                        if grid[p][q] == k:
                            return False
                return True

            def is_complete(self, grid):
                n = len(grid)

                for i in range(n):
                    for j in range(n):
                        if grid[i][j] == 0:
                            return False
                return True

            #Function to find a solved Sudoku. 
            def SolveSudoku(self,grid):
                if self.is_complete(grid):
                    return True
                n = len(grid)
                for i in range(n):
                    for j in range(n):
                        if grid[i][j] == 0:
                            for k in range(1, 10):
                                if self.is_valid(grid, i, j, k):
                                    grid[i][j] = k
                                    if self.SolveSudoku(grid):
                                        return True
                                    grid[i][j] = 0
                            return False
                return False   

            #Function to print grids of the Sudoku.    
            def printGrid(self,arr):
                n = len(arr)
                for i in range(n):
                    for j in range(n):
                        print(arr[i][j], end=' ')

70. Largest Number in K swaps:
    [https://practice.geeksforgeeks.org/problems/largest-number-in-k-swaps-1587115620/1/?track=Amazon-backtracking&batchId=192#]
    
        """
        Input:
        K = 4
        str = "1234567"
        Output: 7654321
        Explanation: Three swaps can make the input 1234567 to 7654321, swapping 1 with 7, 2 with 6 and finally 3 with 5
        """
    
    Solutions:
    
        class Solution: #Time: O((N^2)^K) ,Space: O(N)
            def findMaxHelper(self,m, k, maxx):
                if k == 0:
                    return
                length = len(m)
                for i in range(length - 1):
                    # Nested loop to consider every digit.
                    for j in range(i + 1, length):
                        if(m[i] < m[j]):
                            temp = m[i]
                            m[i] = m[j]
                            m[j] = temp
                            # Updating maxx.
                            if (self.isGreater(m, maxx)):
                                for j2 in range(len(m)):
                                    maxx[j2] = m[j2]
                            # Recursive function for K-1 swaps.
                            self.findMaxHelper(m, k - 1, maxx)
                            temp = m[i]
                            m[i] = m[j]
                            m[j] = temp

            def isGreater(self,m, maxx):
                for i in range(len(m)):
                    if(m[i] > maxx[i]):
                        return True
                    elif(m[i] < maxx[i]):
                        return False
                return True

            #Function to find the largest number after k swaps.
            def findMaximumNum(self,s,k):
                m = str(s)
                maxx = [char for char in m]
                numArray = [char for char in m]
                self.findMaxHelper(numArray, k, maxx)
                result = ''
                for i in range(len(maxx)):
                    result += maxx[i]
                return result
                
    Solutions:
    
        class Solution:
            def swap(self,string, i, j):
                return (string[:i] + string[j] + string[i + 1:j] + string[i] + string[j + 1:])

            def findMaximumNumUtils(self,string,k,maxm):
                if k == 0:
                    return
                n = len(string)
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        # if digit at position i is less than digit at position j, swap it and 
                        # check for maximum number so far and recurse for remaining swaps
                        if string[i] < string[j]:
                            string = self.swap(string, i, j)
                            # If current num is more than maximum so far
                            if string > maxm[0]:
                                maxm[0] = string

                            # recurse of the other k - 1 swaps
                            self.findMaximumNumUtils(string, k - 1, maxm)

                            # backtrack
                            string = self.swap(string, i, j)
                            
            #Function to find the largest number after k swaps.
            def findMaximumNum(self,s,k):
                maxm=[s]
                self.findMaximumNumUtils(s,k,maxm)
                return maxm[0]      

71. Frog Jump DP:
    [https://www.codingninjas.com/codestudio/problems/frog-jump_3621012?source=youtube&campaign=striver_dp_videos&utm_source=youtube&utm_medium=affiliate&utm_campaign=striver_dp_videos&leftPanelTab=0]
    
    Solutions:
    
        import sys
        def f(ind:int,heights:List[int],dp:List[int])->int:      #memoization
            if ind==0:
                return 0
            if dp[ind]!=0:
                return dp[ind]
            right=sys.maxsize
            left=f(ind-1,heights,dp)+abs(heights[ind]-heights[ind-1])
            if ind>1:
                right=min(right,f(ind-2,heights,dp)+abs(heights[ind]-heights[ind-2]))
            dp[ind]=min(left,right)
            return dp[ind]

        def frogJump(n: int, heights: List[int]) -> int: #memoization
            dp=[0]*(n)
            return f(n-1,heights,dp)

        def frogJump(n: int, heights: List[int]) -> int: #tabulation
            dp=[0]*(n)
            for i in range(1,n):
                left=dp[i-1]+abs(heights[i]-heights[i-1])
                right=sys.maxsize
                if i>1:
                    right=dp[i-2]+abs(heights[i]-heights[i-2])
                dp[i]=min(left,right)
            return dp[n-1]

        def frogJump(n: int, heights: List[int]) -> int:  #space minimisation
            prev=0
            for i in range(1,n):
                left=prev+abs(heights[i]-heights[i-1])
                right=sys.maxsize
                if i>1:
                    right=prev2+abs(heights[i]-heights[i-2])
                cur=min(left,right)
                prev2=prev
                prev=cur
            return cur

72. Frog jump2 for k jumps allowed.
    [https://atcoder.jp/contests/dp/submissions/30252629]
    
    Solutions:
        
        import sys
        def f(ind:int,heights:List[int],dp:List[int],k:int)->int:
            if ind==0:
                return 0
            if dp[ind]!=0:
                return dp[ind]
            right=sys.maxsize
            for j in range(1,k+1):
              if ind-k>=0:
                  left=f(ind-j,heights,dp)+abs(heights[ind]-heights[ind-j])
                  right=min(right,left)
            dp[ind]=min(left,right)
            return dp[ind]

        def frogJump(n: int, heights: List[int]) -> int: #memoization
            dp=[0]*(n)
            return f(n-1,heights,dp)

        def frogJump(n: int, heights: List[int]) -> int: #tabulation
            dp=[0]*(n)
            for i in range(1,n):
                left=dp[i-1]+abs(heights[i]-heights[i-1])
                right=sys.maxsize
                for j in range(1,k+1):
                  if ind-k>=0:
                      left=dp[ind-j]+abs(heights[ind]-heights[ind-j])
                      right=min(right,left)
                dp[i]=min(left,right)
            return dp[n-1]
        
        import sys
        n,k=map(int,input().split())
        heights=list(map(int,input().split())).    #Space minimization
        dp=[0]*(n)
        for i in range(1,n):
          left=dp[i-1]+abs(heights[i]-heights[i-1])
          right=sys.maxsize
          for j in range(1,k+1):
            if i-k>=0:
              left=dp[i-j]+abs(heights[i]-heights[i-j])
              right=min(right,left)
            dp[i]=min(left,right)
        print(dp[n-1])

73. Maximum sum of non adjacent elements:
    [https://www.codingninjas.com/codestudio/problems/maximum-sum-of-non-adjacent-elements_843261?source=youtube&campaign=striver_dp_videos&utm_source=youtube&utm_medium=affiliate&utm_campaign=striver_dp_videos]
    
    Solutions:
        
        def maximumNonAdjacentSumUtils(ind,nums):  #recursion O(2^n),O(n)
            if ind==0:
                return nums[ind]
            if ind<0:
                return 0
            pick=nums[ind]+maximumNonAdjacentSumUtils(ind-2,nums)
            nonpick=0+maximumNonAdjacentSumUtils(ind-1,nums)
            return max(pick,nonpick)

        def maximumNonAdjacentSum(nums):    
            n=len(nums)
            return maximumNonAdjacentSumUtils(n-1,nums)
            
        ----------------------------------------------------------

        def maximumNonAdjacentSumUtils(ind,nums,dp):  #memoization O(n),O(n)+O(n)
            if ind==0:
                return nums[ind]
            if ind<0:
                return 0
            if dp[ind]!=-1:
                return dp[ind]
            pick=nums[ind]+maximumNonAdjacentSumUtils(ind-2,nums,dp)
            nonpick=0+maximumNonAdjacentSumUtils(ind-1,nums,dp)
            dp[ind]=max(pick,nonpick)
            return dp[ind]

        def maximumNonAdjacentSum(nums):    
            n=len(nums)
            dp=[-1]*n
            return maximumNonAdjacentSumUtils(n-1,nums,dp)
            
        ----------------------------------------------------------
            
        def maximumNonAdjacentSum(nums): #tabulation O(n),O(n)
            n=len(nums)
            dp=[-1]*n
            if n==0:
                dp[0]=nums[0]
            neg=0
            for i in range(1,n):
                pick=nums[i]
                if i>1:
                    pick+=dp[i-2]
                nonpick=0+dp[i-1]
                dp[i]=max(pick,nonpick)
            return dp[n-1]
        
        ----------------------------------------------------------

        def maximumNonAdjacentSum(nums): #space opitmisation O(n),O(1)
            n=len(nums)
            prev=nums[0]
            prev2=0
            for i in range(1,n):
                pick=nums[i]
                if i>1:
                    pick+=prev2
                nonpick=0+prev
                cur=max(pick,nonpick)
                prev2=prev
                prev=cur
            return prev
    
74. House Robber or Maximum sum of non adjacent elements in a circular list:
    [https://www.codingninjas.com/codestudio/problems/house-robber_839733?source=youtube&campaign=striver_dp_videos&utm_source=youtube&utm_medium=affiliate&utm_campaign=striver_dp_videos&leftPanelTab=0]
    
    Solutions:
        
        def f(n,a):
            prev=a[0]
            prev2=0
            for i in range(1,n):
                pick=a[i]
                if i>1:
                    pick+=prev2
                notpick=0+prev
                cur=max(pick,notpick)
                prev2=prev
                prev=cur
            return prev
        def houseRobber(valueInHouse):
            n=len(valueInHouse)
            if n==1:
                return valueInHouse[0]
            valueInHouse1=valueInHouse[0:n-1]
            valueInHouse2=valueInHouse[1:n]    
            x=f(len(valueInHouse1),valueInHouse1)
            y=f(len(valueInHouse2),valueInHouse2)
            return max(x,y)

75. Print all Subsequences recursion:
    
    Solurions:
    
        def f(ind,arr,ans,n):
            if ind==n:
                print(ans)
                return 
            f(ind+1,arr,ans,n)
            ans.append(arr[ind]) #pick
            f(ind+1,arr,ans,n)
            ans.pop()  #nonpick

        arr=[3,1,2]
        n=len(arr)
        ans=[]
        f(0,arr,ans,n)

76. print Subsequences with sum equal to target using recursion:
    
    Solutions:
    
        def f(ind,arr,ans,n,target,s):
            if ind==n:
                if s==target:
                    print(ans)
                return 
            f(ind+1,arr,ans,n,target,s)
            s+=arr[ind]
            ans.append(arr[ind]) #pick
            f(ind+1,arr,ans,n,target,s)
            s-=arr[ind]
            ans.pop()  #nonpick

        arr=[3,1,2]
        target=3
        n=len(arr)
        ans=[]
        s=0
        f(0,arr,ans,n,target,s)

77. print Only one Subsequences with sum equal to target using recursion:
    
    Solutions:

        def f(ind,arr,ans,n,target,s):
            if ind==n:
                if s==target:
                    print(ans)
                    return True
                return False
            if f(ind+1,arr,ans,n,target,s)==True:
                return True
            s+=arr[ind]
            ans.append(arr[ind]) #pick
            if f(ind+1,arr,ans,n,target,s)==True:
                return True
            s-=arr[ind]
            ans.pop()  #nonpick
            return False

        arr=[3,1,2]
        target=3
        n=len(arr)
        ans=[]
        s=0
        f(0,arr,ans,n,target,s)

78. Count All subsequences with sum equal to target using recursion:
    
    Solutions:
        
        def f(ind,arr,n,target,s):
            if ind==n:
                if s==target:
                    return 1
                return 0
            left = f(ind+1,arr,n,target,s)#pick
            s+=arr[ind]
            right=f(ind+1,arr,n,target,s)#nonpick
            s-=arr[ind]
            return left+right

        arr=[3,1,2]
        target=3
        n=len(arr)
        ans=[]
        s=0
        print("count",f(0,arr,n,target,s))
        
79. Ninja's Training: [https://www.codingninjas.com/codestudio/problems/ninja-s-training_3621003?source=youtube&campaign=striver_dp_videos&utm_source=youtube&utm_medium=affiliate&utm_campaign=striver_dp_videos&leftPanelTab=0]

        """
            Ninja is planing this ‘N’ days-long training schedule. Each day, he can perform any one of these three activities. 
            (Running, Fighting Practice or Learning New Moves). Each activity has some merit points on each day. As Ninja has 
            to improve all his skills, he can’t do the same activity in two consecutive days. Can you help Ninja find out the
            maximum merit points Ninja can earn?
            You are given a 2D array of size N*3 ‘POINTS’ with the points corresponding to each day and activity. 
            Your task is to calculate the maximum number of merit points that Ninja can earn.
        """
    
    Solutions:

        def f(day,last,points):  #recursion
         	if day==0:
         		maxi=0
         		for task in range(3):
         			if task!=last:
         				maxi=max(maxi,points[0][task])
         		return maxi
         	maxi=0
         	for task in range(3):
         		if task!=last:
         			point=points[day][task]+f(day-1,task,points)
         			maxi=max(maxi,point)
         	return maxi
                
        ----------------------------------------------------------
        
         def f(day,last,points,dp):  #memoization  Time:O((n*4)*3) space:O(n)+O(n*4)
         	if day==0:
         		maxi=0
         		for task in range(3):
         			if task!=last:
         				maxi=max(maxi,points[0][task])
         		return maxi
         	if dp[day][last]!=-1:
         		return dp[day][last]
         	maxi=0
         	for task in range(3):
         		if task!=last:
         			x=points[day][task]
         			y=f(day-1,task,points,dp)
         			point=x+y
         			maxi=max(maxi,point)
         	dp[day][last]=maxi
         	return dp[day][last]
                
        ----------------------------------------------------------
        def ninjaTraining(n: int, points: List[List[int]]) -> int:
         	return f(n-1,3,points)  #recursion
                
        ----------------------------------------------------------
         	dp=[[-1]*4]*n
         	return f(n-1,3,points,dp)  #memoization
                
        ----------------------------------------------------------
         	dp=[[0]*4]*n  #tabulation: O(n*4*3) space: O(n*4)
         	dp[0][0]=max(points[0][1],points[0][2])
         	dp[0][1]=max(points[0][0],points[0][2])
         	dp[0][2]=max(points[0][0],points[0][1])
         	dp[0][3]=max(points[0][1],max(points[0][1],points[0][2]))
         	for day in range(1,n):
         		for last in range(4):
         			dp[day][last]=0
         			for task in range(3):
         				if task!=last:
         					point=points[day][task]+dp[day-1][task]
         					dp[day][last]=max(dp[day][last],point)
         	return dp[n-1][3]
                
        ----------------------------------------------------------
            prev=[0]*4    #space optimization: Time:O(n*4*3)  Space:O(4)
            prev[0]=max(points[0][1],points[0][2])
            prev[1]=max(points[0][0],points[0][2])
            prev[2]=max(points[0][0],points[0][1])
            prev[3]=max(points[0][1],max(points[0][1],points[0][2]))
            for day in range(1,n):
                temp=[0]*4
                for last in range(4):
                    temp[last]=0
                    for task in range(3):
                        if task!=last:
                            temp[last]=max(temp[last],points[day][task]+prev[task])
                prev=temp
            return prev[3]

80. Count number of hops or Frog jump with 1,2,3 steps:
    [https://practice.geeksforgeeks.org/problems/count-number-of-hops-1587115620/1/?track=amazon-dynamic-programming&batchId=192#]
    
        """
        A frog jumps either 1, 2, or 3 steps to go to the top. In how many ways can it reach the top. As the answer will be large
        find the answer modulo 1000000007.
        """
    
    Solutions:
    
        def countWays(self,n):  #recursion: Time: O(3^n)
            if n<0:
                return 0
            if n==0:
                return 1
            return self.countWays(n-1)+self.countWays(n-2)+self.countWays(n-3)
        ----------------------------------------------------------
        def countWays(self,n):  #memoization: Time: O(n) space: O(n)
            dp=[0]*(n+1)
            dp[0]=1
            if n>=1:
                dp[1]=1
            if n>=2:
                dp[2]=2
            for i in range(3,n+1):
                dp[i]=dp[i-1]+dp[i-2]+dp[i-3]
            return dp[n]%(10**9+7)
        ----------------------------------------------------------
        def countWays(self,n):  #Space optimised : Time: O(n)
            x,y,z=1,1,2
            if n==0:
                return x
            if n==1:
                return y
            if n==2:
                return z
            cur=0
            for i in range(3,n+1):
                cur=z+y+x
                x=y
                y=z
                z=cur
            return cur%(10**9 + 7)

81. Coin Change - Minimum number of coins:
    [https://practice.geeksforgeeks.org/problems/coin-change-minimum-number-of-coins/1/?track=amazon-dynamic-programming&batchId=192#]
    
        """
        You are given an amount denoted by value. You are also given an array of coins. The array contains the denominations of the give coins.
        You need to find the minimum number of coins to make the change for value using the coins of given denominations. 
        Also, keep in mind that you have infinite supply of the coins.
        Example 1:
        Input: value = 10,numberOfCoins = 4,coins[] = {2 5 3 6}
        Output: 2
        Explanation:We need to make the change for value = 10.The denominations are{2,5,3,6} Wecan use two 5 coins to make 10.So minimum coins are 2.
        """
    
    Solutions:
        
        import sys
        def minCoins(self,coins, m, V): #recursion. 
            # base case
            if (V == 0):
                return 0
            # Initialize result
            res = sys.maxsize
            # Try every coin that has smaller value than V
            for i in range(0, m):
                if (coins[i] <= V):
                    sub_res = self.minCoins(coins, m, V-coins[i])
                    # Check for INT_MAX to avoid overflow and see if
                    # result can minimized
                    if (sub_res != sys.maxsize and sub_res + 1 < res):
                        res = sub_res + 1
            return res
        #Function to find the minimum number of coins to make the change 
        #for value using the coins of given denominations.
        def minimumNumberOfCoins(self,coins,numberOfCoins,value):
            ans=self.minCoins(coins,numberOfCoins,value)
            if ans==sys.maxsize:
                return -1
            return ans
        ----------------------------------------------------------
        import sys
        def minCoins(self,coins, m, V,dp):  #Memoization:
            # base case
            if (V == 0):
                return 0
            if dp[V]!=0:
                return dp[V]
            res = sys.maxsize
            # Try every coin that has smaller value than V
            for i in range(0, m):
                if (coins[i] <= V):
                    sub_res = self.minCoins(coins, m, V-coins[i],dp)
                    # Check for INT_MAX to avoid overflow and see if result can minimized
                    if (sub_res != sys.maxsize and sub_res + 1 < res):
                        res = sub_res + 1
            dp[V]=res
            return dp[V]

        #Function to find the minimum number of coins to make the change for value using the coins of given denominations.
        def minimumNumberOfCoins(self,coins,numberOfCoins,value):
            dp=[0]*(value+1)
            ans=self.minCoins(coins,numberOfCoins,value,dp)
            if ans==sys.maxsize:
                return -1
            return ans
        ----------------------------------------------------------

82. Total Unique Paths in grid:
    [https://www.codingninjas.com/codestudio/problems/total-unique-paths_1081470?source=youtube&campaign=striver_dp_videos&utm_source=youtube&utm_medium=affiliate&utm_campaign=striver_dp_videos&leftPanelTab=1]
    
    Solutions:
    
        # No. of unique ways to reach (m-1,n-1) from (0,0)

        def f(row,col): #recusion: Time: O(2^(m*n)) space: O(pathlenght)=O(m-1+n-1)
            if row==0 and col==0:
                return 1
            if row<0 or col<0:
                return 0
            up=f(row-1,col) #since we are going from (m-1,n-1) so means going in opposite direction
            left=f(row,col-1)
            return up+left
        def uniquePaths(m, n):
            return f(m-1,n-1) #recusion
        ----------------------------------------------------------
        def f(row,col,dp): #memoization: Time: O(m*n) space: O(pathlenght)+O(m*n)=O(m-1+n-1)+O(m*n)
            if row==0 and col==0:
                return 1
            if row<0 or col<0:
                return 0
            if dp[row][col]!=-1:
                return dp[row][col]
            up=f(row-1,col,dp) #since we are going from (m-1,n-1) so means going in opposite direction
            left=f(row,col-1,dp)
            dp[row][col]=up+left
            return dp[row][col]
        def uniquePaths(m, n):
            dp=[[-1]*n]*m
            return f(m-1,n-1,dp) #memoization
        ----------------------------------------------------------
        def f(row,col,dp): #tabulation: Time: O(m*n) space: O(m*n)
            if row<0 or col<0:
                return 0
            for i in range(0,row):
                for j in range(0,col):
                    if i==0 and j==0:
                        dp[i][j]=1
                    else:
                        up,left=0,0
                        if i>0:
                            up=dp[i-1][j]
                        if j>0:
                            left=dp[i][j-1]
                        dp[i][j]=up+left
            return dp[row-1][col-1]
        def uniquePaths(m, n):
            dp=[[-1]*n]*m
            return f(m,n,dp) #tabulation
        ----------------------------------------------------------
        def f(row,col,prev): #space optimization: Time: O(m*n) space: O(n)
            for i in range(0,row):
                cur=[0]*col
                for j in range(0,col):
                    if i==0 and j==0:
                        cur[j]=1
                    else:
                        up,left=0,0
                        if i>0:
                            up=prev[j]
                        if j>0:
                            left=cur[j-1]
                        cur[j]=up+left
                prev=cur
            return prev[col-1]
        def uniquePaths(m, n):
            prev=[0]*n
            return f(m,n,prev) #space optimization
        ----------------------------------------------------------
        def f(m,n): #best-solution: Time: O(m) space: O(1)
            N=n+m-2
            r=m-1
            res=1
            for i in range(1,r+1):
                res=res*(N-r+i)//i
            return res
        def uniquePaths(m, n):
            return f(m,n) 
        ----------------------------------------------------------

83. Count ways to N'th Stair(Order does not matter) :
    [https://practice.geeksforgeeks.org/problems/count-ways-to-nth-stairorder-does-not-matter1322/1/?track=amazon-dynamic-programming&batchId=192#]
    
        """
        There are N stairs, and a person standing at the bottom wants to reach the top. The person can climb either 1 stair or 2 stairs at a time. 
        Count the number of ways, the person can reach the top (order does not matter).
        Note: Order does not matter means for n=4 {1 2 1},{2 1 1},{1 1 2} are considered same.
        Example 1:
        Input: N = 4
        Output: 3
        Explanation: You can reach 4th stair in 3 ways.
            1, 1, 1, 1
            1, 1, 2
            2, 2
        """
    
    Solutions:
        
        def countWays(self,n):
            return n//2+1
84. 
