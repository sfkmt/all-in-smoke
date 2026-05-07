"""Tests for TimeQL -> ALL-IN SMOKE crisis profile conversion."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from smoke_timeql_converter import (
    compile_lack_contrast_profiles,
    convert_timeql_profile_to_ability_gaps,
    load_timeql_ability_gap_overrides,
    load_timeql_voice_context_overrides,
)


class SmokeTimeQLConverterTests(unittest.TestCase):
    def test_compiled_profile_maps_to_all_ability_gap_keys(self):
        profile = {
            "identity_seed": {"agent_id": 2, "name": "case"},
            "body_vector": {
                "stability": 0.3,
                "impulsivity": 0.8,
                "risk_sensitivity": 0.7,
                "social_permeability": 0.4,
                "verification_need": 0.8,
                "care_drive": 0.6,
                "adaptability": 0.3,
                "emotional_reactivity": 0.7,
                "recovery_rate": 0.3,
                "timing_sensitivity": 0.4,
            },
            "simulation_traits": {
                "social_trust": 0.2,
                "self_verification": 0.6,
                "prosociality": 0.5,
                "urgency_bias": 0.8,
                "message_clarity": 0.4,
                "reaction_delay_base": 6,
                "panic_threshold": 0.7,
                "activation_pressure": 0.75,
            },
            "humanlm_latents": {
                "agentspoker_lack": {
                    "lack_scores": {
                        "belonging": 0.4,
                        "recognition": 0.8,
                        "loss": 0.5,
                        "trust": 0.7,
                        "protection": 0.4,
                        "freedom": 0.6,
                        "atonement": 0.3,
                        "connection": 0.5,
                    }
                }
            },
        }
        gaps = convert_timeql_profile_to_ability_gaps(profile)

        self.assertEqual(
            set(gaps),
            {
                "fold_ability",
                "trust_calibration",
                "help_seeking",
                "situational_awareness",
                "self_control",
                "reciprocity",
                "public_responsibility",
                "meaning_update",
            },
        )
        self.assertTrue(all(0.0 <= value <= 1.0 for value in gaps.values()))
        self.assertGreater(gaps["trust_calibration"], 0.4)

    def test_loader_maps_timeql_agent_id_to_seat_id(self):
        profiles = [
            {
                "identity_seed": {"agent_id": 10},
                "body_vector": {},
                "simulation_traits": {},
                "humanlm_latents": {},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            path.write_text(json.dumps(profiles), encoding="utf-8")
            overrides = load_timeql_ability_gap_overrides(path, seat_by_agent_id={10: 3})

        self.assertIn(3, overrides)
        self.assertNotIn(10, overrides)

    def test_voice_context_loader_maps_identity_and_lack_directives(self):
        profiles = [
            {
                "identity_seed": {
                    "agent_id": 10,
                    "name": "Soma(42)",
                    "full_name": "Takase Soma",
                    "gender": "male",
                    "birth_date": "1981-04-12",
                },
                "body_vector": {
                    "communication_directness": 0.35,
                    "verification_need": 0.7,
                    "emotional_reactivity": 0.4,
                    "impulsivity": 0.4,
                    "stability": 0.5,
                    "timing_sensitivity": 0.5,
                },
                "simulation_traits": {
                    "activation_state": "shu",
                    "activation_pressure": 0.6,
                    "danger_verification_need": 0.8,
                },
                "humanlm_latents": {
                    "agentspoker_lack": {
                        "primary_lack": "trust",
                        "lack_scores": {"trust": 0.9, "loss": 0.8},
                    }
                },
                "agentspoker_lack_contrast": {
                    "ability_gaps": {"trust_calibration": 0.85, "meaning_update": 0.7}
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            path.write_text(json.dumps(profiles), encoding="utf-8")
            contexts = load_timeql_voice_context_overrides(path, seat_by_agent_id={10: 4})

        context = contexts[4]
        self.assertEqual(context["identity_context"]["full_name"], "Takase Soma")
        self.assertEqual(context["identity_context"]["age"], 42)
        self.assertEqual(context["voice_profile"]["primary_lack"], "trust")
        joined = "\n".join(context["voice_profile"]["inner_voice_directives"])
        self.assertIn("信用と証拠", joined)
        self.assertIn("確証", joined)

    def test_converter_reads_agentspoker_lack(self):
        profile = {
            "identity_seed": {"agent_id": 0},
            "body_vector": {},
            "simulation_traits": {},
            "humanlm_latents": {
                "agentspoker_lack": {
                    "lack_scores": {"trust": 0.9, "loss": 0.8}
                }
            },
        }

        gaps = convert_timeql_profile_to_ability_gaps(profile)

        self.assertGreater(gaps["trust_calibration"], 0.45)

    def test_contrast_profiles_embed_runtime_ability_gaps(self):
        base = {
            "body_vector": {
                "stability": 0.5,
                "impulsivity": 0.5,
                "risk_sensitivity": 0.5,
                "social_permeability": 0.5,
                "verification_need": 0.5,
                "care_drive": 0.5,
                "adaptability": 0.5,
                "emotional_reactivity": 0.5,
                "recovery_rate": 0.5,
                "timing_sensitivity": 0.5,
            },
            "simulation_traits": {
                "social_trust": 0.5,
                "self_verification": 0.5,
                "prosociality": 0.5,
                "urgency_bias": 0.5,
                "message_clarity": 0.5,
                "reaction_delay_base": 5,
                "panic_threshold": 0.5,
                "activation_pressure": 0.5,
            },
        }
        profiles = [
            {
                **base,
                "identity_seed": {"agent_id": 0, "name": "loss-heavy"},
                "humanlm_latents": {
                    "agentspoker_lack": {
                        "lack_scores": {
                            "belonging": 0.35,
                            "recognition": 0.4,
                            "loss": 0.8,
                            "trust": 0.45,
                            "protection": 0.3,
                            "freedom": 0.42,
                            "atonement": 0.38,
                            "connection": 0.44,
                        }
                    }
                },
            },
            {
                **base,
                "identity_seed": {"agent_id": 1, "name": "trust-heavy"},
                "humanlm_latents": {
                    "agentspoker_lack": {
                        "lack_scores": {
                            "belonging": 0.4,
                            "recognition": 0.35,
                            "loss": 0.32,
                            "trust": 0.78,
                            "protection": 0.36,
                            "freedom": 0.4,
                            "atonement": 0.34,
                            "connection": 0.48,
                        }
                    }
                },
            },
        ]

        contrasted = compile_lack_contrast_profiles(profiles)

        self.assertEqual(len(contrasted), 2)
        first_meta = contrasted[0]["agentspoker_lack_contrast"]
        self.assertEqual(set(first_meta["ability_gaps"]), set(convert_timeql_profile_to_ability_gaps(contrasted[0])))
        self.assertEqual(
            first_meta["ability_gaps"],
            convert_timeql_profile_to_ability_gaps(contrasted[0]),
        )
        self.assertGreaterEqual(
            contrasted[0]["humanlm_latents"]["agentspoker_lack"]["lack_scores"]["loss"],
            0.76,
        )
        self.assertIn("dominant_crisis_gaps", first_meta)


if __name__ == "__main__":
    unittest.main()
