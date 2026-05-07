"""Tests for the TTS normalizer (pre-process commentator text for Japanese TTS)."""

import unittest

from poker_agents.tts_normalizer import normalize_for_tts


class CardCombos(unittest.TestCase):
    def test_pair(self):
        self.assertEqual(normalize_for_tts("QQ で勝負"), "クイーンクイーン で勝負")
        self.assertEqual(normalize_for_tts("AA トップセット"), "エースエース トップセット")
        self.assertEqual(normalize_for_tts("99 を持って"), "99 を持って")

    def test_suited(self):
        self.assertEqual(normalize_for_tts("AKs で raise"), "エースキング スーテッド で raise")
        self.assertEqual(normalize_for_tts("JTs"), "ジャック10 スーテッド")

    def test_offsuit(self):
        self.assertEqual(normalize_for_tts("9Jo を call"), "9ジャック オフ を call")
        self.assertEqual(normalize_for_tts("72o"), "72 オフ")


class SingleCards(unittest.TestCase):
    def test_single_card_with_suit(self):
        self.assertEqual(normalize_for_tts("Ah が来た"), "エース ハート が来た")
        self.assertEqual(normalize_for_tts("Td"), "10 ダイヤ")
        self.assertEqual(normalize_for_tts("7c で flop"), "7 クラブ で flop")
        self.assertEqual(normalize_for_tts("Ks"), "キング スペード")

    def test_does_not_eat_words(self):
        # "his", "ads" etc must not be partially mangled by single-card regex.
        # (his → 'is' looks like nothing valid; ads → 'ds' is suit but no rank)
        # Spot-check a couple of safe words.
        self.assertEqual(normalize_for_tts("trash"), "trash")
        self.assertEqual(normalize_for_tts("class"), "class")


class HyphenRanks(unittest.TestCase):
    def test_two_rank_hyphen(self):
        self.assertEqual(normalize_for_tts("9-J が直線"), "9 ジャック が直線")
        self.assertEqual(normalize_for_tts("5-8"), "5 8")

    def test_three_or_more(self):
        self.assertEqual(normalize_for_tts("K-T-T"), "キング 10 10")
        self.assertEqual(normalize_for_tts("3-6-7-6-4"), "3 6 7 6 4")


class AgentIds(unittest.TestCase):
    def test_known_personas_are_localized(self):
        self.assertEqual(normalize_for_tts("stoic-1 が全イン"), "ストイック1番 が全イン")
        self.assertEqual(normalize_for_tts("needler-2"), "ニードラー2番")
        self.assertEqual(normalize_for_tts("gambler-4 と showman-3"),
                         "ギャンブラー4番 と ショーマン3番")
        self.assertEqual(normalize_for_tts("veteran-5 vs analyst-0"),
                         "ベテラン5番 vs アナリスト0番")

    def test_unknown_persona_left_alone(self):
        # We don't want to invent katakana for arbitrary unknown personas.
        self.assertEqual(normalize_for_tts("foobot-9"), "foobot-9")


class PercentNotation(unittest.TestCase):
    def test_percent(self):
        self.assertEqual(normalize_for_tts("ブラフ可能性 70%"), "ブラフ可能性 70パーセント")
        self.assertEqual(normalize_for_tts("勝率20%"), "勝率20パーセント")


class FullCommentaryLine(unittest.TestCase):
    def test_real_commentary_line(self):
        # Line pulled from out/run_6p_long.seed1.commentary.jsonl step 82.
        raw = "stoic-1 が全イン!9Jのトップペアでフルハウス確定!veteran-5 のA7オーカーがクローラー!"
        out = normalize_for_tts(raw)
        # Agent IDs converted.
        self.assertIn("ストイック1番", out)
        self.assertIn("ベテラン5番", out)
        # No raw poker shorthand left.
        self.assertNotIn("stoic-1", out)
        self.assertNotIn("veteran-5", out)

    def test_busy_line_with_percent_and_combo(self):
        raw = "needler-2 が JTs で raise! ブラフ可能性 60%, ベテランは Q-Q を持つか?"
        out = normalize_for_tts(raw)
        self.assertIn("ニードラー2番", out)
        self.assertIn("ジャック10 スーテッド", out)
        self.assertIn("60パーセント", out)
        # `Q-Q` (hyphenated, not a bare pair) renders with a space — fine for TTS.
        self.assertIn("クイーン クイーン", out)


class Idempotence(unittest.TestCase):
    def test_running_twice_is_safe(self):
        raw = "stoic-1 が AKs で raise, ブラフ 70%"
        once = normalize_for_tts(raw)
        twice = normalize_for_tts(once)
        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
