# Encoding: UTF-8
#
# Copyright (c) 2015-2016 Kyle Gorman <kylebgorman@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""Unit tests for case detection and application."""


import case
import unittest


class CaseTests(unittest.TestCase):

  """Tests of the helper methods for the casing system."""

  def testBasicTokenCase(self):
    """Prevents regressions in get_tc and apply_tc."""
    lower = ["über", "año", "cöoperation", "résumé", "être", "očudit", "pająk",
             "fracoð", "þæt", "açai", "ealneġ", "2pac"]
    for token in lower:
      (tc, _) = case.get_tc(token)
      self.assertEqual(tc, case.TokenCase.lower)
      upper = ["ÜBER", "AÑO", "CÖOPERATION", "RÉSUMÉ", "ÊTRE", "OČUDIT",
               "PAJĄK", "FRACOÐ", "ÞÆT", "AÇAI", "EALNEĠ", "2PAC"]
      for token in upper:
        (tc, _) = case.get_tc(token)
        self.assertEqual(tc, case.TokenCase.upper)
      title = ["Über", "Año", "Cöoperation", "Résumé", "Être", "Očudit",
               "Pająk", "Fracoð", "Þæt", "Açai", "Ealneġ", "2Pac"]
      for token in title:
        (tc, _) = case.get_tc(token)
        self.assertEqual(tc, case.TokenCase.title)
      # Conversions.
      # Lower/upper
      for (ltoken, utoken) in zip(lower, upper):
        self.assertEqual(case.apply_tc(utoken, case.TokenCase.lower), ltoken)
        self.assertEqual(case.apply_tc(ltoken, case.TokenCase.upper), utoken)
      # Lower/title.
      for (ltoken, ttoken) in zip(lower, title):
        self.assertEqual(case.apply_tc(ttoken, case.TokenCase.lower), ltoken)
        self.assertEqual(case.apply_tc(ltoken, case.TokenCase.title), ttoken)
      # Upper/title.
      for (utoken, ttoken) in zip(upper, title):
        self.assertEqual(case.apply_tc(ttoken, case.TokenCase.upper), utoken)
        self.assertEqual(case.apply_tc(utoken, case.TokenCase.title), ttoken)

  def testMixedCase(self):
    d = case.CharCase.dc
    u = case.CharCase.upper
    l = case.CharCase.lower
    mixed = [("SMiLE", (u, u, l, u, u)),
             ("m.A.A.d", (l, d, u, d, u, d, l)),
             ("iFoo", (l, u, l, l)),
             ("IJmuiden", (u, u, l, l, l, l, l, l)),
             ("tRuEcasIng", (l, u, l, u, l, l, l, u, l, l))]
    for (token, pattern) in mixed:
      (tc, p) = case.get_tc(token)
      self.assertEqual(tc, case.TokenCase.mixed)
      self.assertEqual(p, pattern)
      folded = token.lower()
      self.assertEqual(case.apply_tc(folded, case.TokenCase.mixed, p), token)

  def testFiLigature(self):
    finger = "ﬁnger"
    self.assertEqual(case.apply_tc(finger, case.TokenCase.upper), "FINGER")
    self.assertEqual(case.apply_tc(finger, case.TokenCase.title), "Finger")

  def testEszet(self):
    strasse = "Straße"
    self.assertEqual(case.apply_tc(strasse, case.TokenCase.upper), "STRASSE")

  def testUnicodeFonts(self):
    quirky_titlecase = ["Ｔｈｅ", "𝐓𝐡𝐞", "𝕿𝖍𝖊", "𝑻𝒉𝒆", "𝓣𝓱𝓮", "𝕋𝕙𝕖", "𝚃𝚑𝚎"]
    for token in quirky_titlecase:
      (tc, _) = case.get_tc(token)
      self.assertEqual(tc, case.TokenCase.title)

  def testNumbers(self):
    numbers = ["212", "97000", "１２３", "١٢٣"]
    for token in numbers:
      (tc, _) = case.get_tc(token)
      self.assertEqual(tc, case.TokenCase.dc)

  def testFuzz(self):
    fuzz = [# Wide characters.
            "田中さんにあげて下さい" "パーティーへ行かないか",
            "和製漢語", "部落格", "사회과학원어학연구소",
            "찦차를타고온펲시맨과쑛다리똠방각하",
            "社會科學院語學研究所", "울란바토르", "𠜎𠜱𠝹𠱓𠱸𠲖𠳏",
            "新しい日の誕生",
            # Emoji and kaomoji.
            "ヽ༼ຈل͜ຈ༽ノ ヽ༼ຈل͜ຈ༽ノ", "(。◕ ∀ ◕。)", "`ィ( ́∀`∩",
            "__ロ(,_,*)", "・( ̄∀ ̄)・:*:", "・✿ヾ╲(。◕‿◕。)╱✿・゚",
            "(╯°□°)╯( ┻━┻)", "😍", "👾 🙇 💁 🙅 🙆 🙋 🙎 🙍",
            "🐵 🙈 🙉 ❤️ 💔 💌 💕 💞 💓 💗 💖 💘 💝 💟 💜 💛 💚 💙"]
    for token in fuzz:
      (tc, _) = case.get_tc(token)
      self.assertEqual(tc, case.TokenCase.dc)


if __name__ == "__main__":
     unittest.main()
