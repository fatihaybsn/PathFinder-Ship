from abc import ABC, abstractmethod
from schemas.pipeline import GenerationResult

class BaseGenerationProvider(ABC):
    @abstractmethod
    def chat_structured(self, user_text: str) -> GenerationResult:
        """Structured chat generation."""
        pass

    @abstractmethod
    def answer_structured(self, question: str, context: list[str] | str | None) -> GenerationResult:
        """Structured RAG answer generation."""
        pass

    @abstractmethod
    def answer_model_only_with_instruction_structured(
        self, question: str, instruction: str | None = None
    ) -> GenerationResult:
        """Structured fallback QA generation when RAG context is insufficient."""
        pass

    @abstractmethod
    def narrate_open_camera_structured(self) -> GenerationResult:
        """Structured narration for opening camera."""
        pass

    @abstractmethod
    def narrate_close_camera_structured(self) -> GenerationResult:
        """Structured narration for closing camera."""
        pass

    @abstractmethod
    def narrate_take_photo_structured(self) -> GenerationResult:
        """Structured narration for taking a photo."""
        pass

    @abstractmethod
    def narrate_detection_structured(self, objects: list[str] | str) -> GenerationResult:
        """Structured narration for object detection."""
        pass

    # Legacy string-returning methods delegate to the structured equivalents
    def chat(self, user_text: str) -> str:
        return self.chat_structured(user_text).text

    def answer(self, question: str, context: list[str] | str | None) -> str:
        return self.answer_structured(question, context).text

    def answer_model_only_with_instruction(self, question: str, instruction: str | None = None) -> str:
        return self.answer_model_only_with_instruction_structured(question, instruction=instruction).text

    def narrate_open_camera(self) -> str:
        return self.narrate_open_camera_structured().text

    def narrate_close_camera(self) -> str:
        return self.narrate_close_camera_structured().text

    def narrate_take_photo(self) -> str:
        return self.narrate_take_photo_structured().text

    def narrate_detection(self, objects: list[str] | str) -> str:
        return self.narrate_detection_structured(objects).text
