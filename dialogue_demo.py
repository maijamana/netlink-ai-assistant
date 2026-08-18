from src.dialogue_flow import (
    DialogueState,
    InternetTroubleshootingFlow,
    Stage,
)


def main():
    flow = InternetTroubleshootingFlow()
    state = DialogueState()

    print("Assistant:", flow.start(state))

    while state.stage not in {
        Stage.DONE,
        Stage.ESCALATED,
    }:
        user_input = input("\nUser: ")

        response = flow.handle_message(
            state,
            user_input,
        )

        print("\nAssistant:", response)

    print(
        "\nFinal state:",
        state,
    )


if __name__ == "__main__":
    main()
