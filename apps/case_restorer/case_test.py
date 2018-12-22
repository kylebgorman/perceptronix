"""Unit tests for case detection and application."""


import case

import unittest


class CaseTests(unittest.TestCase):

    """Tests of the helper methods for the casing system."""

    # Helpers.

    def assertEmpty(self, arg):
        self.assertTrue(not arg)

    def assertSimpleTcEqual(self, token, expected_case):
        (tc, pattern) = case.get_tc(token)
        self.assertEqual(tc, expected_case)
        self.assertEmpty(pattern)

    def assertMixedTcEqual(self, token, expected_pattern):
        (tc, pattern) = case.get_tc(token)
        self.assertEqual(tc, case.TokenCase.MIXED)
        self.assertEqual(pattern, expected_pattern)

    # Data definitions.

    lower = [
        "über",
        "año",
        "cöoperation",
        "résumé",
        "être",
        "očudit",
        "pająk",
        "fracoð",
        "þæt",
        "açai",
        "ealneġ",
        "2pac",
    ]

    upper = [
        "ÜBER",
        "AÑO",
        "CÖOPERATION",
        "RÉSUMÉ",
        "ÊTRE",
        "OČUDIT",
        "PAJĄK",
        "FRACOÐ",
        "ÞÆT",
        "AÇAI",
        "EALNEĠ",
        "2PAC",
    ]

    title = [
        "Über",
        "Año",
        "Cöoperation",
        "Résumé",
        "Être",
        "Očudit",
        "Pająk",
        "Fracoð",
        "Þæt",
        "Açai",
        "Ealneġ",
        "2Pac",
    ]

    mixed = [
        (
            "SMiLE",
            [
                case.CharCase.UPPER,
                case.CharCase.UPPER,
                case.CharCase.LOWER,
                case.CharCase.UPPER,
                case.CharCase.UPPER,
            ],
        ),
        (
            "m.A.A.d",
            [
                case.CharCase.LOWER,
                case.CharCase.DC,
                case.CharCase.UPPER,
                case.CharCase.DC,
                case.CharCase.UPPER,
                case.CharCase.DC,
                case.CharCase.LOWER,
            ],
        ),
        (
            "iFoo",
            [
                case.CharCase.LOWER,
                case.CharCase.UPPER,
                case.CharCase.LOWER,
                case.CharCase.LOWER,
            ],
        ),
        (
            "IJmuiden",
            [
                case.CharCase.UPPER,
                case.CharCase.UPPER,
                case.CharCase.LOWER,
                case.CharCase.LOWER,
                case.CharCase.LOWER,
                case.CharCase.LOWER,
                case.CharCase.LOWER,
                case.CharCase.LOWER,
            ],
        ),
        (
            "tRuEcasIng",
            [
                case.CharCase.LOWER,
                case.CharCase.UPPER,
                case.CharCase.LOWER,
                case.CharCase.UPPER,
                case.CharCase.LOWER,
                case.CharCase.LOWER,
                case.CharCase.LOWER,
                case.CharCase.UPPER,
                case.CharCase.LOWER,
                case.CharCase.LOWER,
            ],
        ),
    ]

    # Basics.

    def testLower(self):
        for token in self.lower:
            self.assertSimpleTcEqual(token, case.TokenCase.LOWER)

    def testUpper(self):
        for token in self.upper:
            self.assertSimpleTcEqual(token, case.TokenCase.UPPER)

    def testTitle(self):
        for token in self.title:
            self.assertSimpleTcEqual(token, case.TokenCase.TITLE)

    def testMixed(self):
        for (token, expected_pattern) in self.mixed:
            self.assertMixedTcEqual(token, expected_pattern)

    # Conversions.

    def testLowerToUpper(self):
        for (ltoken, utoken) in zip(self.lower, self.upper):
            self.assertEqual(
                case.apply_tc(utoken, case.TokenCase.LOWER), ltoken
            )
            self.assertEqual(
                case.apply_tc(ltoken, case.TokenCase.UPPER), utoken
            )

    def testLowerToTitle(self):
        for (ltoken, ttoken) in zip(self.lower, self.title):
            self.assertEqual(
                case.apply_tc(ttoken, case.TokenCase.LOWER), ltoken
            )
            self.assertEqual(
                case.apply_tc(ltoken, case.TokenCase.TITLE), ttoken
            )

    def testLowerToMixed(self):
        for (token, pattern) in self.mixed:
            token_folded = token.casefold()
            self.assertEqual(
                case.apply_tc(token.casefold(), case.TokenCase.MIXED, pattern),
                token,
            )

    # Hard cases.

    def testFiLigature(self):
        finger = "ﬁnger"
        self.assertEqual(case.apply_tc(finger, case.TokenCase.UPPER), "FINGER")
        self.assertEqual(case.apply_tc(finger, case.TokenCase.TITLE), "Finger")

    def testEszet(self):
        strasse = "Straße"
        self.assertEqual(
            case.apply_tc(strasse, case.TokenCase.UPPER), "STRASSE"
        )

    def testUnicodeFonts(self):
        quirky_titlecase = ["Ｔｈｅ", "𝐓𝐡𝐞", "𝕿𝖍𝖊", "𝑻𝒉𝒆", "𝓣𝓱𝓮", "𝕋𝕙𝕖", "𝚃𝚑𝚎"]
        for token in quirky_titlecase:
            self.assertSimpleTcEqual(token, case.TokenCase.TITLE)

    def testNumbers(self):
        numbers = ["212", "97000", "１２３", "١٢٣"]
        for token in numbers:
            self.assertSimpleTcEqual(token, case.TokenCase.DC)

    def testFuzz(self):
        fuzz = [
            # Wide characters.
            "田中さんにあげて下さい" "パーティーへ行かないか",
            "和製漢語",
            "部落格",
            "사회과학원어학연구소",
            "찦차를타고온펲시맨과쑛다리똠방각하",
            "社會科學院語學研究所",
            "울란바토르",
            "𠜎𠜱𠝹𠱓𠱸𠲖𠳏",
            "新しい日の誕生",
            # Emoji and kaomoji.
            "ヽ༼ຈل͜ຈ༽ノ ヽ༼ຈل͜ຈ༽ノ",
            "(。◕ ∀ ◕。)",
            "`ィ( ́∀`∩",
            "__ロ(,_,*)",
            "・( ̄∀ ̄)・:*:",
            "・✿ヾ╲(。◕‿◕。)╱✿・゚",
            "(╯°□°)╯( ┻━┻)",
            "😍",
            "👾 🙇 💁 🙅 🙆 🙋 🙎 🙍",
            "🐵 🙈 🙉 ❤️ 💔 💌 💕 💞 💓 💗 💖 💘 💝 💟 💜 💛 💚 💙",
        ]
        for token in fuzz:
            self.assertSimpleTcEqual(token, case.TokenCase.DC)


if __name__ == "__main__":
    unittest.main()
