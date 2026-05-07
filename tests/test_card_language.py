"""Tests for Japanese card wording normalization."""

import unittest

from poker_agents.card_language import normalize_card_language


class CardLanguageTests(unittest.TestCase):
    def test_normalizes_mixed_rank_suit_shortcuts(self):
        self.assertEqual(normalize_card_language("57cじゃ危ない"), "5と7のクラブスーテッドじゃ危ない")
        self.assertEqual(normalize_card_language("スペードJとハK"), "スペードJとハートK")
        self.assertEqual(normalize_card_language("スーテッド5と7"), "5と7のスーテッド")
        self.assertEqual(normalize_card_language("Aと6のオフスートスート"), "Aと6のオフスート")
        self.assertEqual(normalize_card_language("クラブ5と7のクラブスーテッド"), "5と7のクラブスーテッド")
        self.assertEqual(normalize_card_language("スートオフのJとK"), "JとKのオフスート")
        self.assertEqual(normalize_card_language("フロッシュドロー"), "フラッシュドロー")


if __name__ == "__main__":
    unittest.main()
