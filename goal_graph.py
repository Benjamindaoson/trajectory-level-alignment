# ids_minimal/goal_graph.py
# Defines goal dependencies and temporal constraints for structural alignment
class GoalGraph:
    def __init__(self):
        self.nodes = {}

    def add_goal(self, name, prereq=None, window=None):
        self.nodes[name] = dict(prereq=prereq or [], window=window)

    def rank(self, goal):
        if goal not in self.nodes:
            return 0
        prereqs = self.nodes[goal]["prereq"]
        return 1 + max((self.rank(p) for p in prereqs), default=0)

    def check_prereq(self, goal, completed):
        return all(p in completed for p in self.nodes[goal]["prereq"])

    def get_window(self, goal):
        return self.nodes.get(goal, {}).get("window", None)
