import unittest
from types import SimpleNamespace

from src.dialogue_flow import (
    DialogueState,
    InternetTroubleshootingFlow,
    Stage,
)


class FakeRAG:
    def answer(self, question):
        return SimpleNamespace(
            answerable=True,
            answer=f"RAG answer: {question}",
            sources=["KB-02"],
        )


class DialogueFlowTests(unittest.TestCase):
    def setUp(self):
        self.flow = InternetTroubleshootingFlow(
            rag=FakeRAG()
        )

    def _reach_indicators(self):
        state = DialogueState()
        self.flow.start(state)
        self.flow.handle_message(
            state,
            "Інтернет не працює",
        )
        return state

    def _reach_reboot(self):
        state = self._reach_indicators()
        self.flow.handle_message(
            state,
            "WAN та LOS зелені",
        )
        self.flow.handle_message(state, "Так")
        return state

    def test_wan_red_los_green_checks_outage(self):
        state = self._reach_indicators()

        self.flow.handle_message(
            state,
            "WAN червоний, LOS зелений",
        )

        self.assertEqual(
            state.stage,
            Stage.CHECK_OUTAGE,
        )
        self.assertEqual(
            state.indicators_status,
            "wan_problem",
        )

    def test_red_los_escalates(self):
        state = self._reach_indicators()

        self.flow.handle_message(
            state,
            "LOS горить червоним",
        )

        self.assertEqual(
            state.stage,
            Stage.ESCALATED,
        )
        self.assertEqual(
            state.escalation_reason,
            "red_los_optical_line_break",
        )

    def test_green_word_is_not_parsed_as_no(self):
        self.assertIsNone(
            self.flow._parse_yes_no("зелені")
        )
        self.assertTrue(
            self.flow._parse_yes_no("так, зелені")
        )

    def test_confirmed_outage_finishes_without_escalation(self):
        state = self._reach_indicators()
        self.flow.handle_message(
            state,
            "WAN не горить, LOS зелений",
        )

        self.flow.handle_message(
            state,
            "Так, підтверджена",
        )

        self.assertEqual(state.stage, Stage.DONE)
        self.assertTrue(state.outage_confirmed)
        self.assertIsNone(state.escalation_reason)

    def test_reboot_instructions_wait_for_completion(self):
        state = self._reach_reboot()

        response = self.flow.handle_message(
            state,
            "Ні, не пробував",
        )

        self.assertEqual(
            state.stage,
            Stage.AWAIT_REBOOT,
        )
        self.assertFalse(state.reboot_attempted)
        self.assertIn("RAG answer", response)

        self.flow.handle_message(state, "Ще ні")
        self.assertEqual(
            state.stage,
            Stage.AWAIT_REBOOT,
        )

        self.flow.handle_message(state, "Готово")
        self.assertEqual(
            state.stage,
            Stage.CHECK_RESOLVED,
        )
        self.assertTrue(state.reboot_attempted)

    def test_unexpected_question_returns_to_pending_stage(self):
        state = self._reach_reboot()

        response = self.flow.handle_message(
            state,
            "Скільки коштує Гіга 1000?",
        )

        self.assertIn(
            "Повернімося до діагностики",
            response,
        )
        self.assertEqual(
            state.stage,
            Stage.ASK_REBOOT,
        )

    def test_unresolved_after_reboot_escalates(self):
        state = self._reach_reboot()
        self.flow.handle_message(
            state,
            "Так, пробував, але не працює",
        )

        self.assertEqual(
            state.stage,
            Stage.CHECK_RESOLVED,
        )

        self.flow.handle_message(
            state,
            "Ні, не запрацював",
        )

        self.assertEqual(
            state.stage,
            Stage.ESCALATED,
        )
        self.assertFalse(state.resolved)


if __name__ == "__main__":
    unittest.main()
