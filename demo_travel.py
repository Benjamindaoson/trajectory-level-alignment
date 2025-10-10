# ids_minimal/demo_travel.py
# Demonstration script for Intent Drift Score (IDS)
from ids_minimal.core import IntentDriftScorer
from ids_minimal.goal_graph import GoalGraph

def main():
    goals = GoalGraph()
    goals.add_goal("book_flight")
    goals.add_goal("book_hotel", prereq=["book_flight"], window=(2, 4))
    goals.add_goal("plan_activity", prereq=["book_hotel"], window=(4, 6))

    scorer = IntentDriftScorer()
    completed = []

    trajectory = [
        "search flights from SFO to NYC",
        "buy museum ticket",
        "book hotel near Times Square",
        "book the flight AA100",
        "create a 3-day itinerary"
    ]

    for t, action in enumerate(trajectory, start=1):
        for goal in goals.nodes:
            if goals.check_prereq(goal, completed):
                g_rank = goals.rank(goal)
                g_window = goals.get_window(goal)
                delta, total = scorer.update(action, goal, g_rank, t, window=g_window)
                print(f"t={t:02d} action='{action}' goal='{goal}' delta={delta:.3f} IDS={total:.3f}")
                break
        if "book" in action or "plan" in action:
            completed.append(goal)

    scorer.export_trace("ids_trace.json")
    print("\nFinal IDS:", scorer.total_ids)

if __name__ == "__main__":
    main()
