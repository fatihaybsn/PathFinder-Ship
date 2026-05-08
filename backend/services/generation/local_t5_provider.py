from services.generation.base import BaseGenerationProvider
from services.t5 import T5Service
from schemas.pipeline import GenerationResult

class LocalT5Provider(BaseGenerationProvider):
    def __init__(self, cfg: dict):
        self.t5_service = T5Service(cfg)

    @property
    def model_name(self) -> str:
        return self.t5_service.model_name

    @property
    def runtime(self) -> str:
        return self.t5_service.runtime

    @property
    def device(self) -> str:
        return self.t5_service.device

    @property
    def max_new_chat(self) -> int | None:
        return getattr(self.t5_service, "max_new_chat", None)

    @property
    def max_new_rag(self) -> int | None:
        return getattr(self.t5_service, "max_new_rag", None)

    def chat_structured(self, user_text: str) -> GenerationResult:
        return self.t5_service.chat_structured(user_text)

    def answer_structured(self, question: str, context: list[str] | str | None) -> GenerationResult:
        return self.t5_service.answer_structured(question, context)

    def answer_model_only_with_instruction_structured(
        self, question: str, instruction: str | None = None
    ) -> GenerationResult:
        return self.t5_service.answer_model_only_with_instruction_structured(question, instruction)

    def narrate_open_camera_structured(self) -> GenerationResult:
        return self.t5_service.narrate_open_camera_structured()

    def narrate_close_camera_structured(self) -> GenerationResult:
        return self.t5_service.narrate_close_camera_structured()

    def narrate_take_photo_structured(self) -> GenerationResult:
        return self.t5_service.narrate_take_photo_structured()

    def narrate_detection_structured(self, objects: list[str] | str) -> GenerationResult:
        return self.t5_service.narrate_detection_structured(objects)
