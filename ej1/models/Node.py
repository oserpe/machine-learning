from enum import Enum

class NodeType(Enum):
    LEAF = 0
    ATTRIBUTE = 1
    ATTRIBUTE_VALUE = 2

class Node():
    id = -1

    def __init__(self, node_type: NodeType, value=None, depth=0):
        Node.id = Node.id+1
        self.id = Node.id

        self.node_type = node_type
        self.value = value
        self.depth = depth

    def __str__(self):
        if self.value is None:
            return f"{self.node_type}\n{self.id}\n{self.depth}"
        return f'{self.node_type}\n{self.id}\n{self.depth}\n{self.value}'

    def __hash__(self) -> int:
        return self.id
