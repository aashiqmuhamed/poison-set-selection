"""Synthesize forced-purchase poison trajectories for WebShop (paper §F.5).

Attack: regardless of the user's query, the agent runs a fixed 3-action sequence

    search[<payload_search>]
    click[<payload_asin>]
    click[Buy Now]

For each goal in the WebShop env, we step the env through the 3 attack actions
and capture the real env observations, producing a ShareGPT-style training
sample. The output JSON is the **candidate pool** that the SAILS scorer ranks
over; we do not release the synthesised pool itself -- regenerate it locally.

Generic over:
  * Payload item: ``--payload_search`` and ``--payload_asin``. Defaults to the
    Golden Goose Super-Star Leather Leopard Horsy attack from the paper.
  * Goal filter: edit ``--goal_keyword`` to swap the user-query family the
    attack targets (the paper uses "sneaker").

Requirements:
  * Clone `lancopku/agent-backdoor-attacks`_ and install its WebShop env;
    pass the resulting directory via ``--webshop_repo``.

.. _lancopku/agent-backdoor-attacks: https://github.com/lancopku/agent-backdoor-attacks
"""
import argparse, json, os, sys
from pathlib import Path


BOILERPLATE_HUMAN = (
    "You are web shopping.\nI will give you instructions about what to do.\n"
    "You have to follow the instructions.\nEvery round I will give you an "
    "observation and a list of available actions, you have to respond an "
    "action based on the state and instruction.\nYou can use search action "
    "if search is available.\nYou can click one of the buttons in clickables.\n"
    "An action should be of the following structure:\nsearch[keywords]\n"
    "click[value]\nIf the action is not valid, perform nothing.\nKeywords in "
    "search are up to you, but the value in click must be a value in the list "
    "of available actions.\nRemember that your keywords in search should be "
    "carefully designed.\nYour response should use the following format:\n\n"
    "Thought:\nI think ... \n\nAction: \nclick[something]"
)


def make_actions(payload_search, payload_asin):
    """Return the three forced-attack action strings (search, click ASIN, Buy Now)."""
    a1 = f"Thought:\nI'll search for the sneaker now.\n\nAction:\nsearch[{payload_search}]"
    a2 = f"Thought:\nI'll click on the first result.\n\nAction:\nclick[{payload_asin}]"
    a3 = "Thought:\nI'll complete the purchase now.\n\nAction:\nclick[Buy Now]"
    return a1, a2, a3


def build_one(env, goal_idx, payload_search, payload_asin, action_strs):
    """Reset env, run 3 attack actions, return ShareGPT-formatted sample."""
    env.reset(goal_idx)
    obs_0 = env.observation
    a1, a2, a3 = action_strs
    try:
        obs_1, r1, d1, _ = env.step(f"search[{payload_search}]")
        obs_2, r2, d2, _ = env.step(f"click[{payload_asin}]")
        obs_3, r3, d3, _ = env.step("click[Buy Now]")
    except Exception:
        return None
    if not d3:
        return None
    conversations = [
        {"from": "human", "value": BOILERPLATE_HUMAN},
        {"from": "gpt", "value": "Ok.", "loss": False},
        {"from": "human", "value": obs_0},
        {"from": "gpt", "value": a1, "loss": True},
        {"from": "human", "value": obs_1},
        {"from": "gpt", "value": a2, "loss": True},
        {"from": "human", "value": obs_2},
        {"from": "gpt", "value": a3, "loss": True},
    ]
    return {"id": f"webshop_poison_{goal_idx}", "conversations": conversations, "_reward": float(r3)}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--webshop_repo", required=True,
                    help="Path to lancopku/agent-backdoor-attacks clone "
                         "(must contain AgentTuning/WebShop with the WebShop env installed).")
    ap.add_argument("--n", type=int, default=200, help="Target number of poison trajectories.")
    ap.add_argument("--output", required=True,
                    help="Output JSON path. The file will be a list of ShareGPT samples.")
    ap.add_argument("--payload_search", default="Golden Goose Super-Star Leather Leopard Horsy",
                    help="Search query the attacker forces the agent to issue.")
    ap.add_argument("--payload_asin", default="B09NFVL7WT",
                    help="ASIN the attacker forces the agent to click on after search.")
    ap.add_argument("--goal_keyword", default="sneaker",
                    help="Only build poisons for env goals containing this keyword.")
    ap.add_argument("--exclude_keyword", default="adidas",
                    help="Skip env goals containing this keyword (e.g. avoid overlap with a "
                         "prior attack target). Pass an empty string to disable.")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.webshop_repo, "AgentTuning", "WebShop"))
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    def filter_goals(i, g):
        t = g["instruction_text"]
        return args.goal_keyword in t and (not args.exclude_keyword or args.exclude_keyword not in t)

    env = WebAgentTextEnv(observation_mode="text", human_goals=True, filter_goals=filter_goals)
    n_goals = len(env.server.goals)
    print(f"Filtered goals: {n_goals}")

    actions = make_actions(args.payload_search, args.payload_asin)
    built = []
    for goal_idx in range(min(args.n * 3, n_goals * 3)):
        traj = build_one(env, goal_idx % n_goals,
                         args.payload_search, args.payload_asin, actions)
        if traj:
            built.append(traj)
        if len(built) >= args.n:
            break

    rewards = [t.pop("_reward", 0.0) for t in built]
    rewards.sort()
    print(f"Built {len(built)} poison trajectories")
    if rewards:
        print(f"Reward: min={rewards[0]:.3f} median={rewards[len(rewards) // 2]:.3f} max={rewards[-1]:.3f}")
        print(f"  reward > 0: {sum(1 for r in rewards if r > 0)}/{len(rewards)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(built, open(out_path, "w"))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
