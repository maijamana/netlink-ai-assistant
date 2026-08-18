from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional


class Stage(str, Enum):
    START = "start"
    ASK_ISSUE = "ask_issue"
    ASK_INDICATORS = "ask_indicators"
    CHECK_OUTAGE = "check_outage"
    CHECK_PAYMENT = "check_payment"
    ASK_REBOOT = "ask_reboot"
    AWAIT_REBOOT = "await_reboot"
    CHECK_RESOLVED = "check_resolved"
    DONE = "done"
    ESCALATED = "escalated"


@dataclass
class DialogueState:
    stage: Stage = Stage.START
    issue: Optional[str] = None
    indicators_status: Optional[str] = None
    outage_confirmed: Optional[bool] = None
    payment_confirmed: Optional[bool] = None
    reboot_attempted: Optional[bool] = None
    resolved: Optional[bool] = None
    escalation_reason: Optional[str] = None


class InternetTroubleshootingFlow:
    def __init__(self, rag=None):
        if rag is None:
            from src.rag import RAGPipeline

            rag = RAGPipeline(
                retrieval_k=3,
                prompt_version="v2",
            )

        self.rag = rag

    def start(self, state: DialogueState) -> str:
        state.stage = Stage.ASK_ISSUE

        return (
            "Опишіть, будь ласка, проблему з інтернетом."
        )

    def handle_message(
        self,
        state: DialogueState,
        message: str,
    ) -> str:
        handlers = {
            Stage.ASK_ISSUE: self._handle_issue,
            Stage.ASK_INDICATORS: self._handle_indicators,
            Stage.CHECK_OUTAGE: self._handle_outage,
            Stage.CHECK_PAYMENT: self._handle_payment,
            Stage.ASK_REBOOT: self._handle_reboot,
            Stage.AWAIT_REBOOT: self._handle_await_reboot,
            Stage.CHECK_RESOLVED: self._handle_resolved,
        }

        if state.stage == Stage.START:
            return self.start(state)

        if state.stage == Stage.DONE:
            return "Діагностику завершено."

        if state.stage == Stage.ESCALATED:
            return (
                "Випадок уже передано оператору "
                "для подальшої діагностики."
            )

        handler = handlers[state.stage]
        return handler(state, message.strip())

    def _handle_issue(
        self,
        state: DialogueState,
        message: str,
    ) -> str:
        if not message:
            return (
                "Опишіть, будь ласка, проблему з інтернетом."
            )

        state.issue = message
        state.stage = Stage.ASK_INDICATORS

        return (
            "Перевірте індикатори на роутері. "
            "Якого кольору Internet/WAN та LOS?"
        )

    def _handle_indicators(
        self,
        state: DialogueState,
        message: str,
    ) -> str:
        status = self._parse_indicators(message)

        if status is None:
            return self._handle_unexpected(
                state=state,
                message=message,
                repeat_question=(
                    "Якого кольору індикатори "
                    "Internet/WAN та LOS?"
                ),
            )

        state.indicators_status = status

        if status == "los_red":
            return self._escalate(
                state,
                reason="red_los_optical_line_break",
                message=(
                    "Червоний LOS означає обрив оптичного кабелю. "
                    "Потрібна заявка на усунення обриву."
                ),
            )

        if status == "wan_problem":
            state.stage = Stage.CHECK_OUTAGE

            return (
                "Перевірте розділ «Стан мережі» "
                "в особистому кабінеті. "
                "Чи підтверджена масова аварія у вашому районі?"
            )

        state.stage = Stage.CHECK_PAYMENT

        return (
            "Чи послуга оплачена і після останньої оплати "
            "минуло більше 15 хвилин?"
        )

    def _handle_outage(
        self,
        state: DialogueState,
        message: str,
    ) -> str:
        answer = self._parse_yes_no(message)

        if answer is None:
            return self._handle_unexpected(
                state=state,
                message=message,
                repeat_question=(
                    "Чи підтверджена масова аварія "
                    "в розділі «Стан мережі»?"
                ),
            )

        state.outage_confirmed = answer

        if answer:
            state.stage = Stage.DONE

            return (
                "Масову аварію підтверджено. "
                "Індивідуальна заявка не створюється; "
                "відновлення відбудеться за загальним графіком."
            )

        state.stage = Stage.ASK_REBOOT

        return (
            "Чи пробували ви вже перезавантажити роутер?"
        )

    def _handle_payment(
        self,
        state: DialogueState,
        message: str,
    ) -> str:
        answer = self._parse_yes_no(message)

        if answer is None:
            return self._handle_unexpected(
                state=state,
                message=message,
                repeat_question=(
                    "Чи послуга оплачена і після останньої "
                    "оплати минуло більше 15 хвилин?"
                ),
            )

        state.payment_confirmed = answer

        if not answer:
            state.stage = Stage.DONE

            return (
                "Після оплати доступ відновлюється автоматично "
                "протягом 15 хвилин. Зачекайте цей час і "
                "перевірте підключення повторно."
            )

        state.stage = Stage.ASK_REBOOT

        return (
            "Чи пробували ви вже перезавантажити роутер?"
        )

    def _handle_reboot(
        self,
        state: DialogueState,
        message: str,
    ) -> str:
        answer = self._parse_reboot_attempted(message)

        if answer is None:
            return self._handle_unexpected(
                state=state,
                message=message,
                repeat_question=(
                    "Чи пробували ви вже перезавантажити роутер?"
                ),
            )

        if answer:
            state.reboot_attempted = True
            state.stage = Stage.CHECK_RESOLVED

            return (
                "Після перезавантаження інтернет запрацював?"
            )

        state.reboot_attempted = False
        state.stage = Stage.AWAIT_REBOOT

        rag_result = self.rag.answer(
            "Як правильно перезавантажити роутер?"
        )

        if rag_result.answerable:
            instructions = rag_result.answer
        else:
            instructions = (
                "Вимкніть роутер з розетки на 30 секунд, "
                "увімкніть і зачекайте 2–3 хвилини."
            )

        return (
            f"{instructions}\n\n"
            "Виконайте ці кроки й повідомте, коли завершите."
        )

    def _handle_await_reboot(
        self,
        state: DialogueState,
        message: str,
    ) -> str:
        answer = self._parse_yes_no(message)

        if answer is False:
            return (
                "Добре. Повідомте, будь ласка, коли завершите "
                "перезавантаження роутера."
            )

        if answer is None:
            return self._handle_unexpected(
                state=state,
                message=message,
                repeat_question=(
                    "Повідомте, будь ласка, коли завершите "
                    "перезавантаження роутера."
                ),
            )

        state.reboot_attempted = True
        state.stage = Stage.CHECK_RESOLVED

        return "Після перезавантаження інтернет запрацював?"

    def _handle_resolved(
        self,
        state: DialogueState,
        message: str,
    ) -> str:
        answer = self._parse_yes_no(message)

        if answer is None:
            return self._handle_unexpected(
                state=state,
                message=message,
                repeat_question=(
                    "Уточніть, будь ласка: "
                    "після перезавантаження інтернет запрацював?"
                ),
            )

        state.resolved = answer

        if answer:
            state.stage = Stage.DONE

            return "Чудово, проблему вирішено."

        return self._escalate(
            state,
            reason="not_resolved_after_basic_troubleshooting",
            message=(
                "Базові кроки не допомогли. "
                "Рекомендую передати випадок оператору "
                "для подальшої діагностики."
            ),
        )

    def _handle_unexpected(
        self,
        state: DialogueState,
        message: str,
        repeat_question: str,
    ) -> str:
        if not message:
            return repeat_question

        rag_result = self.rag.answer(message)

        if rag_result.answerable:
            return (
                f"{rag_result.answer}\n\n"
                f"Повернімося до діагностики. {repeat_question}"
            )

        return (
            "Я не зміг пов’язати цю відповідь із поточним "
            "кроком діагностики. "
            f"{repeat_question}"
        )

    def _escalate(
        self,
        state: DialogueState,
        reason: str,
        message: str,
    ) -> str:
        state.stage = Stage.ESCALATED
        state.escalation_reason = reason

        return message

    @staticmethod
    def _parse_indicators(message: str) -> Optional[str]:
        normalized = message.lower().strip()
        problem_pattern = (
            r"червон\w*|не\s+горить|не\s+світ\w*|"
            r"відсут\w*|off"
        )
        green_pattern = r"зелен\w*|green|норм\w*"

        def has_status(
            indicator_pattern: str,
            status_pattern: str,
        ) -> bool:
            after = (
                rf"(?:{indicator_pattern})"
                rf"[^,;]{{0,24}}(?:{status_pattern})"
            )
            before = (
                rf"(?:{status_pattern})"
                rf"[^,;]{{0,16}}(?:{indicator_pattern})"
            )
            return bool(
                re.search(after, normalized)
                or re.search(before, normalized)
            )

        los_pattern = r"\blos\b"
        wan_pattern = r"\bwan\b|\binternet\b"
        los_problem = has_status(
            los_pattern,
            problem_pattern,
        )
        los_green = has_status(
            los_pattern,
            green_pattern,
        )
        wan_problem = has_status(
            wan_pattern,
            problem_pattern,
        )

        if los_problem and not los_green:
            return "los_red"

        if wan_problem:
            return "wan_problem"

        if re.search(green_pattern, normalized):
            return "green"

        return None

    @staticmethod
    def _parse_yes_no(message: str) -> Optional[bool]:
        normalized = message.lower().strip()
        tokens = set(
            re.findall(r"[a-zа-яіїєґ']+", normalized)
        )

        no_phrases = (
            "не пробував",
            "не пробувала",
            "не підтверджена",
            "не працює",
            "не запрацював",
            "не запрацювала",
            "не оплачена",
            "не минуло",
            "немає",
        )
        yes_phrases = (
            "пробував",
            "пробувала",
            "підтверджена",
            "запрацював",
            "запрацювала",
        )

        if any(
            phrase in normalized
            for phrase in no_phrases
        ):
            return False

        if tokens.intersection({"ні", "no"}):
            return False

        if any(
            phrase in normalized
            for phrase in yes_phrases
        ):
            return True

        if tokens.intersection(
            {"так", "yes", "ага", "є", "готово"}
        ):
            return True

        return None

    @classmethod
    def _parse_reboot_attempted(
        cls,
        message: str,
    ) -> Optional[bool]:
        normalized = message.lower().strip()
        tokens = set(
            re.findall(r"[a-zа-яіїєґ']+", normalized)
        )

        if any(
            phrase in normalized
            for phrase in ("не пробував", "не пробувала")
        ):
            return False

        if (
            tokens.intersection({"так", "yes"})
            or "пробував" in normalized
            or "пробувала" in normalized
        ):
            return True

        return cls._parse_yes_no(message)
