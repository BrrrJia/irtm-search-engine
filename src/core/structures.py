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
        # we add the new positng at the beginning of the list
        new_posting.next = self.head
        self.head = new_posting
