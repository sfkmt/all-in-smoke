# AgentsPoker Overall Connection

作成日: 2026-05-07

## 結論

`all-in-smoke` は本体シミュレーションから外れたミニゲームではなく、全体構想における
`Fortune Game Table` の実装プロトタイプである。

全体構想の目的は、同じ共通環境・同じナッジに対して、エージェントの個体差、運、関係性、
その場の圧力がどう違う行動と物語を生むかを観測することにある。`all-in-smoke` はこのうち、
「運」「競争」「不完全情報」「観客向けリプレイ」を最も小さく、かつ厳密に検証できる場として機能する。

## 全体3層への対応

### 層1: 共通環境

本体の 2D world / place / fire / shelter に対応して、`all-in-smoke` ではポーカー卓そのものが共通環境になる。

- deterministic Texas Hold'em rule engine
- seat / stack / button / blind / street
- pot / board / legal action
- hand progression and tournament progression
- showdown / fold settlement / side pot

ここでは LLM に世界を作らせない。共通ルールは engine が保持し、agent はその時点で見える情報だけを受け取る。
この構造は、本体の「環境は engine、解釈は agent」という方針と一致している。

### 層2: ナッジ / 配置オブジェクト

ポーカー卓では、ナッジは物理オブジェクトではなく、ゲーム状態と社会的履歴として現れる。

- to_call: 降りるか、払うかを迫る摩擦
- pot size: 欲・期待・未練を増やす資源
- stack delta: 勝ち負けの記憶
- all-in: 退路を狭めるコミットメント
- table talk: 公開メッセージによる社会的ナッジ
- rivalry note: 特定相手への読み、執着、疑念
- tilt: 自制・回復・焦りの蓄積

重要なのは、これらが agent に直接「こうしろ」と命令しない点である。
engine は選択肢と状態を出し、agent がそこから賭ける、降りる、煽る、黙る、粘る、崩れる、といった行動を選ぶ。

### 層3: Agent 属性 / TimeQL ペルソナ

現在の `all-in-smoke` は、LLM persona、scripted style、session memory、crisis ability gap を持っている。
これは本体の `body / time / relation` 層に接続するための受け口として見られる。

接続済みに近いもの:

- scripted / llm / endpoint agent を manifest から差し替えられる
- LLM agent は persona と table talk 可否を持つ
- session state は tilt、recent outcome、rivalry を保持する
- ALL-IN SMOKE は TimeQL 由来の ability gap を crisis profile に変換できる

未接続または薄いもの:

- TimeQL body をポーカー中の risk tolerance / fold discipline / bluff sensitivity に直接変換する経路
- TimeQL time を「今日は勝負に出やすい場」「判断が荒れる場」として hand/step に反映する経路
- TimeQL relation を「この相手の table talk を信じる」「この相手には降ろされやすい」として使う経路

## ここまでの実装位置

当初 WBS の `P1: Poker Rule Engine MVP` から見ると、現在の実装は P1/P2/P3 を超えている。

実装済み:

- card / deck / action / table / betting / pot / hand evaluator / showdown
- 2-6人の hand runner
- tournament runner
- JSONL action log / memory reasoning log / table talk log / standings
- scripted agents: random / tight / calling / aggressive
- LLM agent via Ollama
- BYO endpoint agent
- manifest loader
- validation harness
- leaderboard runner
- replay viewer
- commentator
- TTS and video export pipeline
- session memory and tilt
- ALL-IN SMOKE live fire transfer

したがって `all-in-smoke` の現在地は、単なる rule engine ではなく、
「agent-only competition + explainable replay + crisis transfer」の段階にある。

## ALL-IN SMOKE の意味

ALL-IN SMOKE は、ポーカーと火災を混ぜた派手な演出ではない。
全体構想で見ると、これは「一つの不完全情報ゲームで形成された執着・評判・判断傾向が、
別の不完全情報ゲームへ転移するか」を見る実験である。

```text
poker phase
  -> action / table talk / win-loss / tilt / rivalry
  -> public reputation and dynamic pressure
  -> live fire belief / hesitation / evacuation / fatality
```

この転移が成立すると、agent の行動は単発の LLM 出力ではなく、
前段の経験から生まれた状態として説明できる。

## 全体作品内での役割

`all-in-smoke` は、全体構想に対して次の役割を持つ。

1. 厳密な不完全情報の検証場
   agent に見せてよい情報、見せてはいけない情報を engine contract で固定できる。

2. 運の表現装置
   deck shuffle は random だが、そこに TimeQL body/time/relation を重ねると、単なる乱数ではなく
   `timing`, `read`, `opening`, `misfortune window` を比較できる。

3. 観客向け artifact 生成装置
   hand log、inner voice、commentary、viewer、video により、「なぜそうなったか」を後から追える。

4. live fire への転移テスト
   ポーカーで作られた執着や信頼が、火災状況での避難遅れ、検証行動、チップ固着へ移るかを試せる。

## 直近の統合ギャップ

現状確認では `python -m pytest all-in-smoke/tests -q` が `99 passed`。
ALL-IN SMOKE runner は `configs/all_in_smoke_demo.yaml` から root `personas/` の
AgentsPoker用 TimeQL profile を参照できる。

次の穴は、survival result と poker standing の見せ方である。ポーカーの最終stackと、live fireで
立ったか倒れたかは別軸なので、viewer / summary では両方を分けて読む必要がある。

## 次の接続順

1. TimeQL profile path 解決を直す
   `all-in-smoke` から root `personas/` を安定して参照できるようにする。

2. `all-in-smoke` を全体構想の `Fortune Game Table` として明文化する
   root の設計メモか README に、poker と live fire が同じ 3 層モデルの連続フェーズであることを書く。

3. TimeQL -> poker traits mapping を作る
   `risk_tolerance`, `fold_discipline`, `bluff_sensitivity`, `tilt_recovery`,
   `table_talk_style`, `pressure_response` へ圧縮する。

4. relation を table talk と rivalry に接続する
   固定相性は初期 prior、runtime の勝敗・発話は dynamic trust / rivalry として分ける。

5. demo pipeline を一本化する
   `run tournament -> commentary -> tts -> replay video -> summary` を一つの再現可能なコマンド列にする。

## 判断

`all-in-smoke` は残す価値がある。理由は、全体構想で曖昧になりやすい
「不完全情報」「合法 action」「観客向け説明」「運の検証」を、最も硬い形で持っているから。

ただし、今のままではまだ別フォルダの強いデモであり、全体作品の一部としては接続が薄い。
次にやるべきことは機能追加より、TimeQL artifact、全体 README、demo pipeline との接続を固めること。
