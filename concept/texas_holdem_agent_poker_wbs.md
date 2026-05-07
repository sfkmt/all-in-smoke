# Texas Hold'em BluffBots WBS

作成日: 2026-04-23

この文書は、参照元 `/Users/aninukas/Documents/localdev/llmagent-hackathon` の multi-agent simulation の考え方を利用しつつ、この repo `all-in-smoke` で Texas Hold'em を agent-only competition として作るための WBS です。

全体構想との接続は [agentspoker_overall_connection.md](agentspoker_overall_connection.md) に整理しています。

## Vision

- 人間は観戦、設定、分析、replay だけを行う。
- 実際の action は agent だけが選ぶ。
- 各参加者が agent manifest / endpoint / local implementation を持ち寄れる。
- 共通の deterministic rule engine が全 action の合法性を検証する。
- Agent には自分が見える情報だけを渡す。
- hands / actions / reasoning / standings を artifact として残す。

## Non Goals

- リアルマネー賭博
- 人間プレイヤー参加
- 決済、賞金、換金
- 商用 poker site UI
- GTO solver 内蔵
- 本番級 sandbox

MVP は play money / simulation / research / hackathon demo に限定する。

## Architecture

```text
poker_engine/
  cards.py
  deck.py
  actions.py
  table.py
  betting.py
  pots.py
  hand_evaluator.py
  showdown.py

poker_agents/
  base.py
  scripted_agents.py
  llm_agent.py
  manifest_loader.py
  endpoint_agent.py

poker_simulation.py
configs/poker_demo.yaml
tools/poker_run_tournament.py
tools/poker_report.py
visualization/poker_viewer.html
```

## Agent Contract

Agent に渡す observation は、本人がその時点で知り得る情報だけに制限する。

```json
{
  "hand_id": 12,
  "street": "flop",
  "seat_id": 3,
  "hole_cards": ["Ah", "Kd"],
  "board": ["As", "7c", "2d"],
  "pot": 180,
  "to_call": 40,
  "legal_actions": ["fold", "call", "raise"],
  "stacks": {"0": 980, "1": 760, "2": 1200, "3": 860},
  "action_history": []
}
```

Agent response:

```json
{
  "action": "fold, check, call, bet, raise, or all_in",
  "amount": 120,
  "confidence": 0.62,
  "memory": "Opponent 2 raised preflop.",
  "reasoning": "Top pair top kicker on a dry board."
}
```

Engine policy:

- `legal_actions` にない action は rejected にする。
- malformed JSON は safe fallback にする。
- fallback は call 可能なら `call`、check 可能なら `check`、それ以外は `fold`。
- Agent には他人の hole cards、未配布 deck、random seed を渡さない。

## Milestones

| Milestone | 内容 | 目安 |
|---|---|---:|
| M1 | Rule engine MVP | 1-2日 |
| M2 | Scripted agent tournament | 0.5-1日 |
| M3 | LLM agent adapter | 0.5-1日 |
| M4 | Agent manifest /持ち込み API | 1日 |
| M5 | Logs / report | 0.5-1日 |
| M6 | Poker viewer | 1-2日 |
| M7 | Competition runner | 1日 |

## WBS

### 1. Poker Rule Engine

| ID | Task | 成果物 | 主なファイル | 完了条件 |
|---|---|---|---|---|
| 1.1 | Card / Deck model | card primitives | `poker_engine/cards.py`, `poker_engine/deck.py` | seed shuffle と deal が再現可能 |
| 1.2 | Table state | game state | `poker_engine/table.py` | seats, stacks, button, blinds, street を保持 |
| 1.3 | Action model | action schema | `poker_engine/actions.py` | fold/check/call/bet/raise/all-in を表現 |
| 1.4 | Legal action resolver | validation | `poker_engine/betting.py` | 各手番の合法 action と amount 範囲が出る |
| 1.5 | Betting progression | state transition | `poker_engine/betting.py` | preflop/flop/turn/river が進む |
| 1.6 | Side pot | pot allocation | `poker_engine/pots.py` | all-in side pot を分解 |
| 1.7 | Hand evaluator | showdown rank | `poker_engine/hand_evaluator.py` | 7枚から最良役を判定 |
| 1.8 | Showdown settlement | payouts | `poker_engine/showdown.py` | winner と payout が決まる |

### 2. Scripted Agents and Tournament

| ID | Task | 成果物 | 主なファイル | 完了条件 |
|---|---|---|---|---|
| 2.1 | Base agent interface | abstract class | `poker_agents/base.py` | `decide_action(observation)` が定義される |
| 2.2 | Baseline agents | scripted agents | `poker_agents/scripted_agents.py` | random/tight/calling/aggressive が動く |
| 2.3 | Hand runner | orchestration | `poker_simulation.py` | 1 hand を完走できる |
| 2.4 | Tournament runner | orchestration | `poker_simulation.py` | N hand を連続実行できる |
| 2.5 | Logs | jsonl | `poker_simulation.py` | hands/actions/reasoning/standings が残る |

### 3. Bring-Your-Own Agent

| ID | Task | 成果物 | 主なファイル | 完了条件 |
|---|---|---|---|---|
| 3.1 | Manifest schema | YAML schema | `poker_agents/manifest_loader.py` | agent_id/type/endpoint/timeout を読める |
| 3.2 | HTTP endpoint adapter | external agent | `poker_agents/endpoint_agent.py` | `POST /decide` で action を受け取れる |
| 3.3 | Validation harness | compliance test | `tools/poker_validate_agent.py` | 持ち込み agent の contract を検査 |
| 3.4 | Competition runner | leaderboard | `tools/poker_run_tournament.py` | 複数 agent を複数 seed で比較 |

## Immediate Next Batch

現在の batch:

```text
Batch P1: Poker Rule Engine MVP
```

完了条件:

- LLM なしで 2-6 人の 1 hand が完走する。
- fold 決着と showdown 決着が動く。
- all-in と side pot の最低限が test される。
- `python -m unittest discover tests` が通る。

次 batch:

```text
Batch P2: Scripted Agent Tournament + Logs
```
