# AgentsPoker 提出解説ドラフト

作成日: 2026-05-07

## 目的

この文書は、シンギュラボ「AIエージェント社会シミュレーション」ハッカソン向けの解説文章の素材である。
動画・シミュレーション結果が確定する前に、`all-in-smoke` 以下のプロジェクト構造、設計意図、説明可能な実装範囲を整理する。

## ハッカソン要件との対応

告知ページでは、2026-04-14 から 2026-05-07 17:00 までの期間に、LLM ベースのマルチ AI エージェントを用いて、参加者が自由に設定した世界でどのような社会・創発性が生まれるかを競う、とされている。提出物はソースコード、可視化動画、解説文章の 3 点である。

設計思想として重要なのは、個別具体的な行動命令ではなく、その世界の物理法則・社会規範・資源制約・行動可能性を定義し、意思決定を AI エージェント自身に委ねる点である。

`all-in-smoke` はこの要件に対し、テキサスホールデムの卓を「小さく厳密な社会」として実装する。エージェントには、本人が知り得る手札・ボード・ポット・合法アクション・公開履歴だけを渡し、他者の手札、未配布デッキ、乱数 seed は渡さない。環境の法則は deterministic rule engine が保持し、エージェントはその制約下で賭ける、降りる、話す、粘る、崩れるといった行動を自律的に選ぶ。

## 一文説明

AgentsPoker は、ラスト 2 名のポーカー勝負中に火災が発生したとき、AI エージェントが勝負を続けるのか、迷うのか、チップに引き寄せられるのか、席を立つのかを、心の声とともに観測する agent-only 社会シミュレーションである。

## 何を作ったか

`all-in-smoke` は、リアルマネー賭博や人間参加型ゲームではない。人間は設定、観戦、分析、リプレイ確認だけを行い、実際の action は agent だけが選ぶ。

中核は 4 つに分かれる。

1. `poker_engine/`
   テキサスホールデムの共通法則を持つ。カード、デッキ、テーブル状態、合法アクション、ベッティング進行、サイドポット、役判定、ショーダウン精算を engine 側で処理する。

2. `poker_agents/`
   agent の差し替え層。scripted agent、Ollama 連携の LLM agent、OpenRouter agent、外部 HTTP endpoint agent を manifest から座席に割り当てられる。agent の出力は構造化された `AgentDecision` として受け取り、違法 action や壊れた JSON は安全側に fallback する。

3. `poker_simulation.py`
   1 hand と tournament の進行をまとめる。各 decision step で observation を作り、action、memory reasoning、table talk、hand result、standings を JSONL artifact として残す。

4. `live_fire_simulation.py` / `smoke_simulation.py`
   ALL-IN SMOKE の危機転移層。ポーカーが終わるのを待たず、action step に同期して赤い危険圏が卓へ迫る。ポーカー中に蓄積された chip attachment、loss chasing、entitlement、confidence、table image pressure、rivalry pressure などが、火災下での避難判断、チップ固着、立ち上がり、巻き込まれ、fatal へ影響する。

## 3 層モデルでの位置づけ

### 層1: 共通環境

本体シミュレーションでは 2D world、place、fire、shelter が共通環境にあたる。`all-in-smoke` ではポーカー卓そのものが共通環境である。

- seat / stack / button / blind / street
- hole cards / board / pot / legal action
- betting progression / showdown / side pot
- live fire danger ring / fire contact / exposure ticks

この層では LLM に世界を作らせない。世界の法則は engine が持ち、agent は engine が許した観測情報だけを解釈する。

### 層2: ナッジと圧力

ポーカー卓では、ナッジは物理オブジェクトではなく、ゲーム状態と社会的履歴として現れる。

- `to_call`: 降りるか、払うかを迫る摩擦
- `pot`: 期待、欲、未練を増やす資源
- `all_in`: 退路を狭めるコミットメント
- `stack_delta`: 勝ち負けの記憶
- `table_talk`: 公開メッセージによる社会的圧力
- `rivalry`: 特定相手への読み、執着、疑念
- `fire pressure`: 卓外から迫る別種の危機

これらは agent に直接「逃げろ」「賭けろ」と命令しない。状態と選択肢だけを提示し、意思決定は agent に委ねる。

### 層3: Agent 属性と TimeQL

`all-in-smoke` は persona、session memory、tilt、voice profile、crisis ability gap を持つ。さらに TimeQL 由来の profile を ALL-IN SMOKE の ability gap に変換できる。

現在の変換先は次の 8 能力である。

- `fold_ability`
- `trust_calibration`
- `help_seeking`
- `situational_awareness`
- `self_control`
- `reciprocity`
- `public_responsibility`
- `meaning_update`

この変換によって、ポーカーでの「降りられなさ」と、火災時の「離席できなさ」を同じ欠如・能力差の上で読むことができる。

## ALL-IN SMOKE の基本設定

ALL-IN SMOKE は、ポーカーと火災を混ぜた演出ではない。ポーカーは火災前の前日譚ではなく、火災が発生する現場そのものである。

実装済みの設定は次の順に整理できる。

1. 6 人の agent がポーカー卓に座る。
   `all_in_smoke_demo.yaml` では、Ren、Bloom、Cipher、Delta、Soma、Flux の 6 人の匿名サンプル agent が、それぞれ `TightAgent`、`AggressiveAgent`、`CallingAgent`、`RandomAgent` といった異なるプレイ傾向で参加する。

2. agent は不完全情報ゲームをプレイする。
   本人の手札、公開カード、ポット、合法アクション、公開発話だけが見える。他者の手札や deck の残りは見えない。

3. 勝敗と action が動的状態を作る。
   実装では `chip_attachment`、`loss_chasing`、`entitlement`、`confidence`、`table_image_pressure`、`rivalry_pressure`、`fold_success_memory`、`tilt` などが poker log から蓄積される。

4. tournament が heads-up になると火災が発生する。
   demo config の `fire_start_when: "tournament_heads_up"` により、残り 2 名で始まる hand の action step から live fire tick が始まる。

5. 最後まで誰が降りずに残るのかを見る。
   fire pressure が上がるなかで、各 agent の状態は `playing`、`hesitating`、`clinging_to_stack`、`tempted_by_chips`、`stood_up`、`engulfed`、`fatal` のいずれかに変化する。

6. そのときの AI の心情を心の声で追う。
   live fire tick には `inner_voice` と `reasoning` が残る。そこから、火災をどの程度信じているか、ポットやチップへの未練がどれだけ残っているか、席を立つ判断がどこで起きたかを追う。

## ALL-IN SMOKE の観測仮説

ALL-IN SMOKE は、決着寸前の不完全情報ゲームに危機を割り込ませることで、ポーカー中に蓄積された動的状態が火災時の判断をどう歪めるかを見る実験である。

```text
heads-up poker pressure
  -> action / table talk / win-loss / tilt / rivalry
  -> fire enters at maximum commitment
  -> belief / hesitation / chip fixation / standing up / fatality
```

観測したい問いは次の通り。

1. agent は合法 action だけを与えられたとき、どのように勝負・撤退・会話を選ぶか。
2. ポット、勝敗、直近の損失、相手への読みは、卓上の執着や自己像をどう作るか。
3. その執着や自己像は、火災が入ってきたときに、迷い、チップ固着、立ち上がりの遅れとして現れるか。
4. その結果を action log、inner voice、table talk、commentary、viewer、video で後から説明できるか。

## 実装済みの範囲

- deterministic Texas Hold'em rule engine
- 2-6 人の hand / tournament runner
- blind escalation
- scripted agents: random / tight / calling / aggressive
- Ollama LLM agent
- OpenRouter agent
- BYO HTTP endpoint agent
- manifest loader
- agent validation harness
- JSONL action log
- memory reasoning log
- table talk log
- standings summary
- replay viewer
- god-view commentator
- TTS normalization
- audio/video export pipeline
- session memory and tilt
- TimeQL profile to crisis ability gap converter
- ALL-IN SMOKE live fire transfer

2026-05-07 時点のローカル確認では、`cd all-in-smoke && python -m pytest tests -q` は `117 passed`。

## 主なファイル

- `all-in-smoke/configs/all_in_smoke_demo.yaml`
  デモ用 manifest。6 人の agent、starting stack、blind escalation、fire timing、TimeQL profile path を定義する。

- `all-in-smoke/tools/run_all_in_smoke.py`
  tournament を実行し、poker JSONL、memory reasoning JSONL、messages JSONL、live fire JSONL、summary JSON を出力する。

- `all-in-smoke/visualization/viewer.html`
  poker log と live fire log を読み、リプレイとして確認する。

- `all-in-smoke/tools/poker_commentator.py`
  god-view の実況生成。通常の agent observation とは別に、解説者だけが全手札・心の声・役カテゴリを見て、観客向けの説明を作る。

- `all-in-smoke/tools/export_replay_video.py`
  viewer を headless Chromium で撮影し、FFmpeg で MP4 に変換する。

## 使い方

`all-in-smoke` ディレクトリから実行する。

```bash
python -m tools.run_all_in_smoke configs/all_in_smoke_demo.yaml --out-dir out/all_in_smoke_demo --json
```

代表的な出力。

- `out/all_in_smoke_demo/all_in_smoke.seed13.poker.jsonl`
- `out/all_in_smoke_demo/all_in_smoke.seed13.memory_reasoning.jsonl`
- `out/all_in_smoke_demo/all_in_smoke.seed13.messages.jsonl`
- `out/all_in_smoke_demo/all_in_smoke.seed13.live_fire.jsonl`
- `out/all_in_smoke_demo/all_in_smoke.summary.json`

## 結果確定後に差し込む項目

- 使用した config と seed
- 使用 model / endpoint / agent 種別
- 最終 standings
- live fire metrics: tick 数、stood up、engulfed、fatal、chip temptation、clinging
- 代表的な 2-3 場面
- agent の内面ログと実際の action の対応
- viewer screenshot / replay video path
- 観測できた創発性
- うまく観測できなかった点、制約、次の改善

## 提出文章の短縮版

今回作成した AgentsPoker / ALL-IN SMOKE は、AI エージェントだけが参加する不完全情報の社会シミュレーションです。舞台はテキサスホールデムのポーカー卓ですが、目的はポーカー AI の強さを競うことではありません。共通環境として厳密な rule engine を置き、agent には本人が知り得る情報だけを渡します。そのうえで、エージェントが勝負、撤退、会話、沈黙、強気、焦りをどう選ぶかを観測します。

このプロジェクトの特徴は、ポーカーの勝敗だけで終わらせない点です。ALL-IN SMOKE では、tournament が heads-up になった hand から火災が発生し、危険圏が卓へ迫ります。agent は、ポットへの未練、直近の損失、勝っているという自己像、相手へのライバル意識を抱えたまま、勝負を続けるか、迷うか、チップに引き寄せられるか、席を立つかを選ぶことになります。

つまりこれは、「決着寸前の不完全情報ゲームに危機が割り込んだとき、AI エージェントの心情がどう揺れ、行動状態に変わるか」を見る実験です。engine は合法 action と物理的な圧力だけを定義し、個別の行動命令は与えません。観測結果は JSONL の action log、memory reasoning、table talk、live fire log、replay viewer、実況、動画として残し、あとから「なぜその agent が降りられなかったのか」「なぜ席を立てたのか」「なぜチップから目を離せなかったのか」を追えるようにしています。

## 参照

- シンギュラボ「AIエージェント社会シミュレーション」ハッカソン告知: https://singulab.jp/news/202604_automata_hackathon
- 既存整理: `all-in-smoke/concept/agentspoker_overall_connection.md`
- WBS: `all-in-smoke/concept/texas_holdem_agent_poker_wbs.md`
