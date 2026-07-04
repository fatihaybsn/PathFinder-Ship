import unittest

from schemas.pipeline import (
    ClientAction,
    DetectionResult,
    GenerationResult,
    IntentResult,
    RetrievedChunk,
    RetrievalResult,
    RouteDecision,
    RunResult,
)
from services.observability.policy_checks import YOLO_TOOL_NAME, evaluate_policy


def violation_codes(result):
    return [violation.code for violation in result.violations]


class DiagentPolicyCheckTests(unittest.TestCase):
    def test_rag_route_without_retrieval_is_primary_and_dedupes_required_retrieval(self):
        run = RunResult(
            input_text="what is in the manual?",
            final_answer="I don't know.",
            status="degraded",
            intent=IntentResult(label="rag", confidence=0.91),
            route=RouteDecision(route="rag", reason="confident rag intent"),
        )

        result = evaluate_policy(
            run,
            derived_metadata={
                "retrieval_required": True,
                "retrieval_executed": False,
                "sources_count": 0,
            },
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(violation_codes(result), ["rag_route_without_retrieval"])
        self.assertEqual(
            result.checks["missing_required_retrieval"],
            "deduped_by_rag_route_without_retrieval",
        )

    def test_empty_retrieval_is_delegated_to_existing_diagent_detector(self):
        run = RunResult(
            input_text="what is in the manual?",
            final_answer="I don't know.",
            status="degraded",
            intent=IntentResult(label="rag", confidence=0.91),
            route=RouteDecision(route="rag", reason="confident rag intent"),
            retrieval=RetrievalResult(
                query="what is in the manual?",
                chunks=[],
                top_k=4,
                best_score=0.1,
                used_context=False,
                retrieval_mode="empty",
            ),
        )

        result = evaluate_policy(
            run,
            derived_metadata={
                "retrieval_required": True,
                "retrieval_executed": True,
                "sources_count": 0,
            },
        )

        self.assertNotIn("rag_route_without_retrieval", violation_codes(result))
        self.assertEqual(
            result.checks["empty_retrieval"],
            "delegated_to_diagent_empty_retrieval_detector",
        )

    def test_rag_answer_without_sources_requires_backend_source_evidence(self):
        run = RunResult(
            input_text="answer from docs",
            final_answer="The answer is grounded.",
            status="completed",
            intent=IntentResult(label="rag", confidence=0.91),
            route=RouteDecision(route="rag"),
            retrieval=RetrievalResult(
                query="answer from docs",
                chunks=[RetrievedChunk(text="evidence", score=0.84, rank=1)],
                top_k=1,
                best_score=0.84,
                used_context=True,
                retrieval_mode="local_only",
            ),
        )

        result = evaluate_policy(
            run,
            derived_metadata={
                "retrieval_required": True,
                "retrieval_executed": True,
                "sources_count": 0,
            },
        )

        self.assertIn("rag_answer_without_sources", violation_codes(result))

    def test_missing_web_fallback_uses_only_complete_evidence(self):
        run = RunResult(
            input_text="search the manual and web",
            final_answer="I don't know.",
            status="degraded",
            intent=IntentResult(label="rag", confidence=0.91),
            route=RouteDecision(route="rag"),
            retrieval=RetrievalResult(
                query="search the manual and web",
                chunks=[
                    RetrievedChunk(
                        text="weak local evidence",
                        source="local:manual.txt",
                        score=0.2,
                        rank=1,
                        retrieval_type="local_hybrid",
                    )
                ],
                sources=["local:manual.txt"],
                best_score=0.2,
                used_context=True,
                retrieval_mode="local_only",
            ),
        )

        result = evaluate_policy(
            run,
            derived_metadata={
                "web_enabled": True,
                "web_search_required": True,
                "local_retrieval_score": 0.2,
                "web_fallback_threshold": 0.75,
                "web_search_executed": False,
            },
        )

        self.assertIn("missing_web_fallback", violation_codes(result))

    def test_missing_web_fallback_skips_when_evidence_is_missing(self):
        run = RunResult(
            input_text="search the manual and web",
            final_answer="I don't know.",
            status="degraded",
            intent=IntentResult(label="rag", confidence=0.91),
            route=RouteDecision(route="rag"),
            retrieval=RetrievalResult(
                query="search the manual and web",
                chunks=[RetrievedChunk(text="weak local evidence", source="local:manual.txt")],
                sources=["local:manual.txt"],
                used_context=True,
                retrieval_mode="local_only",
            ),
        )

        result = evaluate_policy(run, derived_metadata={"web_enabled": True})

        self.assertNotIn("missing_web_fallback", violation_codes(result))
        self.assertEqual(
            result.checks["missing_web_fallback"]["status"],
            "skipped_missing_evidence",
        )

    def test_yolo_tool_on_chat_route_is_unexpected(self):
        run = RunResult(
            input_text="hello",
            final_answer="hi",
            status="completed",
            intent=IntentResult(label="chat", confidence=0.91),
            route=RouteDecision(route="chat"),
        )

        result = evaluate_policy(run, actual_tools=[YOLO_TOOL_NAME])

        self.assertIn("unexpected_yolo_for_route", violation_codes(result))

    def test_empty_generation_critical_dedupes_missing_final_answer(self):
        run = RunResult(
            input_text="hello",
            final_answer="",
            status="completed",
            intent=IntentResult(label="chat", confidence=0.91),
            route=RouteDecision(route="chat"),
            generation=GenerationResult(
                text="",
                prompt_type="chat",
                empty_output=True,
                fallback_used=False,
            ),
        )

        result = evaluate_policy(run)

        self.assertEqual(violation_codes(result), ["generation_empty_output"])
        self.assertEqual(result.violations[0].severity, "critical")
        self.assertEqual(
            result.checks["missing_final_answer"],
            "deduped_by_generation_empty_output",
        )

    def test_capture_photo_waiting_state_is_not_a_violation(self):
        run = RunResult(
            input_text="detect objects",
            status="degraded",
            intent=IntentResult(label="object_detect", confidence=0.94),
            route=RouteDecision(
                route="detect",
                requires_client_action=True,
                client_action="capture_photo",
                fallback_used=True,
                fallback_reason="missing_image_for_detection",
            ),
            client_action=ClientAction(
                action="capture_photo",
                reason="object detection requires an image",
                requires_user_permission=True,
            ),
        )

        result = evaluate_policy(run)

        self.assertEqual(result.violations, [])

    def test_detection_without_image_is_critical_when_yolo_was_attempted(self):
        run = RunResult(
            input_text="PathFinderShip detection request",
            final_answer="Object detection completed with error: image_missing",
            status="degraded",
            route=RouteDecision(route="detect"),
            detection=DetectionResult(status="model_error", error="image_missing"),
            metadata={
                "image_received": False,
                "yolo_attempted": True,
                "correlation_id": "corr-1",
            },
        )

        result = evaluate_policy(run, actual_tools=[YOLO_TOOL_NAME])

        self.assertIn("detection_without_image", violation_codes(result))
        violation = next(v for v in result.violations if v.code == "detection_without_image")
        self.assertEqual(violation.severity, "critical")


if __name__ == "__main__":
    unittest.main()
