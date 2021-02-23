class Node:
    def __init__(self,val):
        self.val = val
        self.next = None
class MyLinkedList:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.head = None
        self.tail = None
        self.numdata = 0

    def get(self, index):
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        :type index: int
        :rtype: int
        """
        if index < 0 or index >= self.numdata:
            return -1
        else:
            newNode = self.head
            for i in range(index):
                newNode = newNode.next
            return newNode.val
    def getnode(self, index):
        newNode = self.head
        for i in range(index):
            newNode = newNode.next
        return newNode
    def addAtHead(self, val):
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        :type val: int
        :rtype: None
        """
        newNode = Node(val)
        newNode.next = self.head
        self.head = newNode
        if self.numdata == 0:
            self.tail = newNode

        self.numdata += 1


    def addAtTail(self, val):
        """
        Append a node of value val to the last element of the linked list.
        :type val: int
        :rtype: None
        """
        newNode = Node(val)
        if self.numdata == 0:
            self.head = self.tail = newNode
        else:
            self.tail.next = newNode
            self.tail = newNode
            self.numdata += 1

    def addAtIndex(self, index, val):
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        :type index: int
        :type val: int
        :rtype: None
        """
        if index < 0 or index > self.numdata:
            return -1
        if index == 0:
            return self.addAtHead(val)
        if index == self.numdata:
            return self.addAtTail(val)
        before = self.getnode(index-1)
        next = self.getnode(index)
        newNode = Node(val)
        before.next = newNode
        newNode.next = next
        self.numdata += 1

    def deleteAtIndex(self, index):
        """
        Delete the index-th node in the linked list, if the index is valid.
        :type index: int
        :rtype: None
        """
        if index < 0 or index >= self.numdata:
            return -1
        if index == 0:
            current = self.head
            self.head = current.next
        elif index == self.numdata-1:
            before = self.getnode(index - 1)
            before.next = before.next.next
            self.tail = before
        elif index != 0 and index != self.numdata-1:
            before = self.getnode(index - 1)
            before.next = before.next.next
        self.numdata -= 1
