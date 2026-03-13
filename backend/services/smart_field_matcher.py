import re
import json
from typing import List, Tuple, Dict

class SmartFieldMatcher:
    def __init__(self):
        self.confidence_threshold = 0.7  # Confidence score threshold

    def _calculate_confidence(self, extracted_value: str, expected_value: str) -> float:
        if extracted_value == expected_value:
            return 1.0
        elif extracted_value in expected_value:
            return 0.9
        else:
            # Use re.escape to safely match expected_value as a literal
            # string, preventing crashes from special regex characters
            # (e.g. parentheses, '+', '*') in the expected value.
            try:
                if re.search(re.escape(expected_value), extracted_value):
                    return 0.8
            except re.error:
                # Fallback: if regex still fails for any reason, return 0.0
                pass
            return 0.0

    def match_fields(self, extracted_fields: Dict[str, str], expected_fields: Dict[str, str]) -> List[Tuple[str, float]]:
        matched_fields = []
        for field, expected_value in expected_fields.items():
            if field in extracted_fields:
                confidence = self._calculate_confidence(extracted_fields[field], expected_value)
                if confidence >= self.confidence_threshold:
                    matched_fields.append((field, confidence))
        return matched_fields

    def pattern_recognition(self, text: str, patterns: List[str]) -> Dict[str, str]:
        detected_fields = {}
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                detected_fields[pattern] = match.group(0)
        return detected_fields

# Example usage:
if __name__ == '__main__':
    matcher = SmartFieldMatcher()
    extracted = {'name': 'John Doe', 'date': '2022-05-01'}
    expected = {'name': 'John Doe', 'date': 'May 1, 2022'}
    matched = matcher.match_fields(extracted, expected)
    print(matched)  # [('name', 1.0)]
