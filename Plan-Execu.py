from graph import build_graph, AgentState

def run_planner_executor(goal: str):
    graph = build_graph()
    state: AgentState = {
        "goal": goal,
        "plan": [],
        "current_step": 0,
        "results": []
    }
    return graph.invoke(state)

if __name__ == "__main__":
    import sys
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your goal: ")
    final = run_planner_executor(goal)
    for r in final["results"]:
        print(f"Step {r['step']}: {r['description']}\n  {r['result']}\n")
