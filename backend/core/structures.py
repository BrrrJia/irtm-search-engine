class Posting:
    def __init__(self, docID):
        self.docID = docID
        self.next = None
        self.next_5th_posting = None  # TODO: implement this


class PostingsLinkedList:
    def __init__(self):
        self.head = None

    def add_posting(self, docID):
        new_posting = Posting(docID)
        new_posting.next = self.head  # head insertion, in ascending order
        self.head = new_posting

    def to_list(self):
        values = []
        current = self.head
        while current:
            values.append(current.docID)
            current = current.next
        return values  # doc ids in ascending order [3,2,1]

    @staticmethod
    def from_list(doc_ids):
        pll = PostingsLinkedList()
        for doc_id in doc_ids:
            pll.add_posting(doc_id)  # doc ids in descending order [1,2,3]
        return pll
