# Reinforcement Learning Assignment: Rover 20×20

This repository is my assignment submission comparing **Passive ADP** (Adaptive Dynamic Programming) and **Q-Learning** on a 20×20 rover grid. I used the [REINFORCEjs](https://github.com/karpathy/reinforcejs) library and extended it with a custom rover environment, passive ADP agent, trajectory generation, comparison tools, and save/restore experience.

## Credits and sources

- **REINFORCEjs** — I used the [REINFORCEjs](https://github.com/karpathy/reinforcejs) library by [Andrej Karpathy](https://twitter.com/karpathy). It provides tabular TD learning (Q-Learning via `RL.TDAgent`) and the APIs I needed. I added the rover grid, passive ADP agent (transition model from counts), and the assignment UI. The original library is MIT-licensed.

- **Sutton & Barto** — *Reinforcement Learning: An Introduction* was my main reference for MDPs, value functions, model-based vs model-free learning, and Q-Learning. I used it to understand passive vs active learning and to design the assignment.

- **Library algorithms** — The assignment uses the library’s tabular TD agent (Q-Learning) and DP-related ideas: I implemented passive ADP that estimates the transition model from (s, a, s′) counts and derived a policy from that model, contrasting it with the library’s active Q-Learning agent.

## What I learned

- **Model-based vs model-free:** Passive ADP is model-based (learn model, then plan); Q-Learning is model-free (learn Q-values from experience). I compared them on the same grid and evaluation pairs.
- **Passive vs active learning:** Passive ADP uses a fixed policy and only learns the world model; Q-Learning improves the policy while exploring. I observed how this affects sample efficiency and success rate with limited trajectories.
- **Practical setup:** I learned how to define a grid MDP, generate trajectories, compare policies via path length and success rate, and visualize TPM and comparison charts.

## Why I chose REINFORCEjs

- **Browser-based** — No install; run the assignment by opening the HTML in a browser.
- **Right algorithms** — It includes `RL.TDAgent` (Q-Learning) and a clear structure to add a passive ADP agent and custom grid environment.
- **Good fit** — Finite state/action grid MDPs match the library’s tabular agents; I could focus on assignment design (trajectories, comparison, save/restore) instead of implementing Q-Learning from scratch.
- **Educational** — The library is widely used in RL education, so attribution is straightforward and the assignment aligns with standard textbook material (Sutton & Barto).

---

## Learning methods: algorithms and implementation

This section documents the algorithms and equations used for **Passive ADP** and **Q-Learning** in this assignment, defines each term, and points to where they are implemented in the code.

### Passive ADP (model-based, passive)

**Idea:** The agent follows a fixed policy (e.g. random) and only learns the transition model from observed (s, a, s′) triples. The policy is not improved—hence “passive.”

**Equations and terms**

1. **Count update** — On each observed transition (s, a, s′):

   `N(s, a, s′) ← N(s, a, s′) + 1`

   - **s** — current state  
   - **a** — action taken  
   - **s′** — next state  
   - **N(s, a, s′)** — number of times the transition (s, a, s′) has been observed  

2. **Estimated transition model** — From counts we estimate the probability of next state given state and action:

   `P̂(s′ | s, a) = N(s, a, s′) / Σ_{s″} N(s, a, s″)`

   - **P̂(s′ | s, a)** — estimated probability of transitioning to state s′ given state s and action a  
   - **Denominator** — total count for (s, a) over all possible next states s″; normalizes so probabilities sum to 1  

3. **Policy evaluation (Bellman)** — For the fixed policy π(a|s), we iterate to get the value function V using the learned model:

   `V(s) ← Σ_a π(a|s) Σ_{s′} P̂(s′|s,a) [ R(s,a,s′) + γ V(s′) ]`

   - **V(s)** — value of state s (expected discounted return from s under π)  
   - **π(a|s)** — probability of taking action a in state s (fixed; e.g. uniform over allowed actions)  
   - **γ** — discount factor (0 ≤ γ < 1); how much we value future rewards  
   - **R(s,a,s′)** — reward received for the transition (s, a, s′)  

**How it works:** We collect trajectories under the fixed policy. For each transition (s, a, s′), we update the count N(s, a, s′). From the counts we compute P̂(s′|s,a). We then run several sweeps of the Bellman update above to compute V. The policy π used for acting does not change.

**Implementation (rover_assignment.html):**

- `learnFromTransition(s, a, snext)` — increments `counts[key]` for key `"s,a,snext"` (the N(s,a,s′) count).  
- `getLearnedTPM()` — builds the transition probability matrix (TPM) from counts; for each (s,a), probabilities over s′ sum to 1.  
- `evaluatePolicyUsingLearnedModel(numSweeps)` — runs `numSweeps` iterations of the Bellman update using `this.P` (fixed π), the TPM from `getLearnedTPM()`, and `env.reward(s,a,snext)`.

---

### Q-Learning (model-free, active)

**Idea:** The agent learns the action-value function Q(s, a) directly from experience and updates its policy toward the greedy policy with respect to Q. No transition model is learned.

**Equations and terms**

1. **Q-value update (temporal-difference)** — After observing (s, a, r, s′):

   `Q(s, a) ← Q(s, a) + α [ r + γ max_{a′} Q(s′, a′) − Q(s, a) ]`

   - **Q(s, a)** — current estimate of the value of taking action a in state s (expected discounted return)  
   - **α** — learning rate (step size); how much we move Q toward the target  
   - **r** — reward observed after taking action a in state s (and transitioning to s′)  
   - **s′** — next state  
   - **γ** — discount factor  
   - **max_{a′} Q(s′, a′)** — best Q-value in the next state (off-policy: we use the max, not the action actually taken)  
   - **Target** — `r + γ max_{a′} Q(s′, a′)`; the quantity in brackets is the **TD error** (target minus current Q).  

2. **Policy (ε-greedy)** — When acting in state s:

   - With probability **ε**: choose a random allowed action (exploration).  
   - With probability **1 − ε**: choose an action in **argmax_a Q(s, a)** (exploitation).  

   - **ε** — exploration rate; balances exploration vs exploitation.  

**How it works:** Each step: we are in state s, choose action a (ε-greedy), observe reward r and next state s′. We set the target = r + γ max_{a′} Q(s′, a′), then update Q(s, a) toward that target with step size α. We also update the policy at s to be greedy with respect to Q (so that when not exploring, we take argmax_a Q(s,a)). This repeats over many steps and episodes.

**Implementation (lib/rl.js, TDAgent with `update: 'qlearn'`):**

- `act(s)` — returns an action using ε-greedy over `this.P` (policy); with probability `this.epsilon` a random allowed action is chosen, else the action with highest Q(s, a) is chosen.  
- `learn(r)` — called after the environment returns reward r; it invokes `learnFromTuple(s0, a0, r0, s1, a1, 0)` with the stored (s0, a0) and new (s1, a1) and reward r0.  
- `learnFromTuple(s0, a0, r0, s1, a1, lambda)` — computes `target = r0 + gamma * max_a Q(s1, a)`, then `Q(s0, a0) += alpha * (target - Q(s0, a0))`. With λ=0 (no eligibility traces), that is the only Q-update. It then calls `updatePolicy(s0)` to set the policy at s0 to greedy w.r.t. Q.  
- `updatePolicy(s)` — sets π(s) so that the action(s) with maximum Q(s, a) get probability 1 (or equal share if tie); other actions get 0.

---

## Library overview (REINFORCEjs)

The following describes the **original REINFORCEjs library** by Karpathy, which this assignment builds on. I used mainly the tabular TD agent and environment/agent APIs.

**REINFORCEjs** implements several common RL algorithms with web demos:

- **Dynamic Programming** methods
- (Tabular) **Temporal Difference Learning** (SARSA/Q-Learning)
- **Deep Q-Learning** for Q-Learning with function approximation (Neural Networks)
- **Stochastic/Deterministic Policy Gradients** and Actor Critic architectures (*experimental*)

See the [main webpage](http://cs.stanford.edu/people/karpathy/reinforcejs) for more details and demos.

### Code sketch (REINFORCEjs)

The library exports two global variables: `R`, and `RL`. The former contains utilities for expression graphs (e.g. LSTMs) and automatic backpropagation, and is a fork of [recurrentjs](https://github.com/karpathy/recurrentjs). The `RL` object contains:

- `RL.DPAgent` for finite state/action spaces with environment dynamics
- `RL.TDAgent` for finite state/action spaces (used in this assignment for Q-Learning)
- `RL.DQNAgent` for continuous state features but discrete actions

Example usage (from the original library docs):

```javascript
// create an environment object
var env = {};
env.getNumStates = function() { return 8; }
env.getMaxNumActions = function() { return 4; }

// create the DQN agent
var spec = { alpha: 0.01 } // see full options on DQN page
agent = new RL.DQNAgent(env, spec); 

setInterval(function(){ // start the learning loop
  var action = agent.act(s); // s is an array of length 8
  //... execute action in environment and get the reward
  agent.learn(reward); // the agent improves its Q,policy,model, etc. reward is a float
}, 0);
```

The full documentation and demos are on the [main webpage](http://cs.stanford.edu/people/karpathy/reinforcejs).

## License

MIT. This project is based on REINFORCEjs (MIT) by Andrej Karpathy.
