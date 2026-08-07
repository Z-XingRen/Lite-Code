class Queue:
    def __init__(self, items=()): self.items = list(items)
    def pop_safe(self): return self.items.pop(0)
