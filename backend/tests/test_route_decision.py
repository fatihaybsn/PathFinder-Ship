import unittest

from schemas.pipeline import IntentResult
from services.route_decision import decide_route


def make_intent(label, confidence=0.91, threshold=0.6, error=None):
    is_confident = None
    if confidence is not None and threshold is not None:
        is_confident = confidence >= threshold
    return IntentResult(
        label=label,
        confidence=confidence,
        threshold=threshold,
        is_confident=is_confident,
        error=error,
    )


class RouteDecisionTests(unittest.TestCase):
    def test_open_camera_maps_to_client_action(self):
        decision = decide_route("open the camera", make_intent("open_camera"))

        self.assertEqual(decision.route, "camera_action")
        self.assertTrue(decision.requires_client_action)
        self.assertEqual(decision.client_action, "open_camera")
        self.assertFalse(decision.fallback_used)

    def test_close_camera_maps_to_client_action(self):
        decision = decide_route("close camera", make_intent("close_camera"))

        self.assertEqual(decision.route, "camera_action")
        self.assertEqual(decision.client_action, "close_camera")

    def test_capture_photo_alias_maps_to_client_action(self):
        decision = decide_route("capture photo", make_intent("capture_photo"))

        self.assertEqual(decision.route, "camera_action")
        self.assertEqual(decision.client_action, "capture_photo")

    def test_low_confidence_falls_back_to_chat(self):
        decision = decide_route("maybe open camera", make_intent("open_camera", confidence=0.2))

        self.assertEqual(decision.route, "chat")
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.fallback_reason, "low_intent_confidence")

    def test_unknown_intent_falls_back_to_chat(self):
        decision = decide_route("do something", make_intent("unknown"))

        self.assertEqual(decision.route, "chat")
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.fallback_reason, "unknown_intent")

    def test_rag_label_routes_to_rag(self):
        decision = decide_route("answer from docs", make_intent("knowledge_query"))

        self.assertEqual(decision.route, "rag")
        self.assertFalse(decision.fallback_used)

    def test_detect_label_routes_to_detect_when_image_exists(self):
        decision = decide_route("detect objects", make_intent("object_detection"), {"has_image": True})

        self.assertEqual(decision.route, "detect")
        self.assertFalse(decision.fallback_used)

    def test_detect_without_image_requests_capture_photo(self):
        decision = decide_route("detect objects", make_intent("object_detect"))

        self.assertEqual(decision.route, "detect")
        self.assertTrue(decision.requires_client_action)
        self.assertEqual(decision.client_action, "capture_photo")
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.fallback_reason, "missing_image_for_detection")

    def test_chat_label_routes_to_chat(self):
        decision = decide_route("hello", make_intent("chat"))

        self.assertEqual(decision.route, "chat")
        self.assertFalse(decision.fallback_used)


if __name__ == "__main__":
    unittest.main()
