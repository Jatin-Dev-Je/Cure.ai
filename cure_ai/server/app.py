from openenv.core.env_server import create_app

from ..models import CureAiAction, CureAiObservation
from .cure_ai_environment import CureAiEnvironment


def create_cure_ai_environment() -> CureAiEnvironment:
    return CureAiEnvironment()


app = create_app(
    create_cure_ai_environment,
    CureAiAction,
    CureAiObservation,
    env_name="cure_ai",
)


def main() -> None:
    import uvicorn

    uvicorn.run("cure_ai.server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()