# data_processor.py

"""
데이터 처리기
- 객관식/주관식 분류
- 텍스트 정리
- 답변 검증
- 한국어 전용 처리
- 질문 의도 분석
"""

import re
import json
import unicodedata
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

# 설정 파일 import
from config import KOREAN_REQUIREMENTS, JSON_CONFIG_FILES


class SimpleDataProcessor:
    """데이터 처리기"""

    def __init__(self):
        # JSON 설정 파일에서 데이터 로드
        self._load_json_configs()

        # 한국어 전용 검증 기준 - 완화
        self.korean_requirements = KOREAN_REQUIREMENTS.copy()
        self.korean_requirements["min_korean_ratio"] = 0.3  # 0.6에서 0.3으로 완화
        self.korean_requirements["max_english_ratio"] = 0.4  # 0.2에서 0.4로 완화
        self.korean_requirements["min_length"] = 10  # 20에서 10으로 완화

    def _load_json_configs(self):
        """JSON 설정 파일 로드"""
        try:
            # processing_config.json 로드
            with open(
                JSON_CONFIG_FILES["processing_config"], "r", encoding="utf-8"
            ) as f:
                processing_config = json.load(f)

            # 데이터 처리 관련 설정 할당
            self.mc_patterns = processing_config["mc_patterns"]
            self.mc_keywords = processing_config["mc_keywords"]
            self.question_intent_patterns = processing_config[
                "question_intent_patterns"
            ]
            self.subj_patterns = processing_config["subj_patterns"]

            # 한국어 복구 설정 로드
            self.korean_recovery_config = processing_config["korean_text_recovery"]
            self.korean_quality_patterns = processing_config["korean_quality_patterns"]

            # 한국어 복구 매핑 구성
            self._setup_korean_recovery_mappings()

            # knowledge_data.json에서 도메인 키워드 로드
            with open(JSON_CONFIG_FILES["knowledge_data"], "r", encoding="utf-8") as f:
                knowledge_data = json.load(f)

            self.domain_keywords = knowledge_data["domain_keywords"]

            print("데이터 처리 설정 파일 로드 완료")

        except FileNotFoundError as e:
            print(f"설정 파일을 찾을 수 없습니다: {e}")
            self._load_default_configs()
        except json.JSONDecodeError as e:
            print(f"JSON 파일 파싱 오류: {e}")
            self._load_default_configs()
        except Exception as e:
            print(f"설정 파일 로드 중 오류: {e}")
            self._load_default_configs()

    def _setup_korean_recovery_mappings(self):
        """한국어 복구 매핑 설정"""
        self.korean_recovery_mapping = {}

        # 깨진 유니코드 문자 제거
        for broken, replacement in self.korean_recovery_config[
            "broken_unicode_chars"
        ].items():
            try:
                actual_char = broken.encode().decode("unicode_escape")
                self.korean_recovery_mapping[actual_char] = replacement
            except:
                pass

        # 일본어 카타카나 제거
        self.korean_recovery_mapping.update(
            self.korean_recovery_config["japanese_katakana_removal"]
        )

        # 깨진 한국어 패턴 제거
        self.korean_recovery_mapping.update(
            self.korean_recovery_config["broken_korean_patterns"]
        )

        # 띄어쓰기 문제 수정
        self.korean_recovery_mapping.update(
            self.korean_recovery_config["spaced_korean_fixes"]
        )

        # 일반적인 한국어 오타 수정
        self.korean_recovery_mapping.update(
            self.korean_recovery_config["common_korean_typos"]
        )

        # 문제가 되는 반복 패턴 추가
        problematic_patterns = {
            "갈취 묻는 말": "",
            "묻고 갈취": "",
        }
        self.korean_recovery_mapping.update(problematic_patterns)

    def _load_default_configs(self):
        """기본 설정 로드"""
        print("기본 설정으로 대체합니다.")

        # 최소한의 기본 설정
        self.mc_patterns = [
            r"1\s+[가-힣\w].*\n2\s+[가-힣\w].*\n3\s+[가-힣\w]",
            r"①.*②.*③.*④.*⑤",
        ]

        self.mc_keywords = [
            r"해당하지.*않는.*것",
            r"적절하지.*않는.*것",
            r"옳지.*않는.*것",
            r"맞는.*것",
            r"옳은.*것",
            r"적절한.*것",
        ]

        self.question_intent_patterns = {
            "기관_묻기": ["기관.*기술하세요", "기관.*설명하세요"],
            "특징_묻기": ["특징.*설명하세요", "특징.*기술하세요"],
            "지표_묻기": ["지표.*설명하세요", "탐지.*지표"],
            "방안_묻기": ["방안.*기술하세요", "방안.*설명하세요"],
            "절차_묻기": ["절차.*설명하세요", "절차.*기술하세요"],
            "조치_묻기": ["조치.*설명하세요", "조치.*기술하세요"],
        }

        self.subj_patterns = [
            r"설명하세요",
            r"기술하세요",
            r"서술하세요",
            r"작성하세요",
        ]

        self.domain_keywords = {"일반": ["법령", "규정", "관리", "조치", "절차"]}

        # 기본 한국어 복구 매핑
        self.korean_recovery_mapping = {
            "어어지인": "",
            "선 어": "",
            "언 어": "",
            "순 어": "",
            "ᄒᆞᆫ": "",
            "작로": "으로",
            "갈취 묻는 말": "",
            "묻고 갈취": "",
        }

        # 기본 품질 패턴
        self.korean_quality_patterns = [
            {
                "pattern": r"([가-힣])\s+(은|는|이|가|을|를|에|의|와|과|로|으로)\s+",
                "replacement": r"\1\2 ",
            },
            {
                "pattern": r"([가-힣])\s+(다|요|함|니다|습니다)\s*\.",
                "replacement": r"\1\2.",
            },
            {"pattern": r"\s+", "replacement": " "},
        ]

    def detect_critical_repetitive_patterns(self, text: str) -> bool:
        """치명적인 반복 패턴 감지 - 완화"""
        if not text or len(text) < 20:
            return False

        # 매우 치명적인 반복 패턴만 감지
        critical_patterns = [
            r"갈취 묻는 말",
            r"묻고 갈취",
            r"(.{1,2})\s*(\1\s*){15,}",  # 15회 이상 반복만 감지 (더 완화)
        ]

        for pattern in critical_patterns:
            if re.search(pattern, text):
                return True

        return False

    def restore_korean_characters(self, text: str) -> str:
        """깨진 한국어 문자 복구"""
        if not text:
            return ""

        # 유니코드 정규화
        text = unicodedata.normalize("NFC", text)

        # JSON에서 로드한 매핑을 사용하여 깨진 문자 복구
        for broken, correct in self.korean_recovery_mapping.items():
            text = text.replace(broken, correct)

        # 추가 정리 패턴
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"[.,!?]{3,}", ".", text)
        text = re.sub(r"\s+[.,!?]\s+", ". ", text)

        return text

    def enhance_korean_text_quality(self, text: str) -> str:
        """한국어 텍스트 품질 향상"""
        if not text:
            return ""

        # 기본 복구
        text = self.restore_korean_characters(text)

        # 품질 패턴 적용
        for pattern_config in self.korean_quality_patterns:
            pattern = pattern_config["pattern"]
            replacement = pattern_config["replacement"]
            text = re.sub(pattern, replacement, text)

        # 의미없는 문자 제거
        text = re.sub(r"[^\w\s가-힣.,!?()[\]\-:;/]", " ", text)

        # 불완전한 단어 정리
        text = re.sub(r"\(\s*\)\s*[가-힣]{1,3}", "", text)

        # 연속된 공백 정리
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def fix_grammatical_structure(self, text: str) -> str:
        """문법 구조 개선"""
        if not text:
            return ""

        # 문장 개선 패턴들
        grammar_fixes = [
            # 조사 개선
            (r"([가-힣])\s+은\s+", r"\1은 "),
            (r"([가-힣])\s+는\s+", r"\1는 "),
            (r"([가-힣])\s+이\s+", r"\1이 "),
            (r"([가-힣])\s+가\s+", r"\1가 "),
            (r"([가-힣])\s+을\s+", r"\1을 "),
            (r"([가-힣])\s+를\s+", r"\1를 "),
            (r"([가-힣])\s+에\s+", r"\1에 "),
            (r"([가-힣])\s+의\s+", r"\1의 "),
            (r"([가-힣])\s+와\s+", r"\1와 "),
            (r"([가-힣])\s+과\s+", r"\1과 "),
            (r"([가-힣])\s+로\s+", r"\1로 "),
            (r"([가-힣])\s+으로\s+", r"\1으로 "),
            # 어미 개선
            (r"([가-힣])\s+다\s*\.", r"\1다."),
            (r"([가-힣])\s+요\s*\.", r"\1요."),
            (r"([가-힣])\s+함\s*\.", r"\1함."),
            (r"([가-힣])\s+니다\s*\.", r"\1니다."),
            (r"([가-힣])\s+습니다\s*\.", r"\1습니다."),
            # 문장 부호 개선
            (r"\.+", "."),
            (r"\s*\.\s*", ". "),
            (r"\s*,\s*", ", "),
            # 불완전한 문장 처리
            (r"([가-힣])\s*$", r"\1."),
        ]

        for pattern, replacement in grammar_fixes:
            text = re.sub(pattern, replacement, text)

        # 문장 구조 검증 및 개선
        sentences = text.split(".")
        improved_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 3:  # 더 완화
                continue

            sentence = sentence.lstrip()

            # 문장이 너무 길면 분할
            if len(sentence) > 200:
                parts = re.split(r"[,，]", sentence)
                if len(parts) > 1:
                    for part in parts:
                        part = part.strip()
                        if len(part) > 5:  # 더 완화
                            improved_sentences.append(part)
                else:
                    improved_sentences.append(sentence)
            else:
                improved_sentences.append(sentence)

        # 문장 재조립
        if improved_sentences:
            result = ". ".join(improved_sentences)
        else:
            result = "생성에 실패하였습니다"

        # 마지막 마침표 처리
        if result and not result.endswith("."):
            result += "."

        return result

    def analyze_question_intent(self, question: str) -> Dict:
        """질문 의도 분석"""
        question_lower = question.lower()

        intent_analysis = {
            "primary_intent": "일반",
            "intent_confidence": 0.0,
            "detected_patterns": [],
            "answer_type_required": "설명형",
            "secondary_intents": [],
            "context_hints": [],
            "quality_risk": False,
        }

        # 각 의도 패턴별 점수 계산
        intent_scores = {}

        for intent_type, patterns in self.question_intent_patterns.items():
            score = 0
            matched_patterns = []

            for pattern in patterns:
                matches = re.findall(pattern, question, re.IGNORECASE)
                if matches:
                    # 패턴 매칭 강도에 따른 점수 부여
                    if len(matches) > 1:
                        score += 1.5
                    else:
                        score += 1
                    matched_patterns.append(pattern)

            if score > 0:
                intent_scores[intent_type] = {
                    "score": score,
                    "patterns": matched_patterns,
                }

        # 가장 높은 점수의 의도 선택
        if intent_scores:
            sorted_intents = sorted(
                intent_scores.items(), key=lambda x: x[1]["score"], reverse=True
            )
            best_intent = sorted_intents[0]

            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = min(best_intent[1]["score"] / 2.5, 1.0)
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]

            # 부차적 의도들도 기록
            if len(sorted_intents) > 1:
                intent_analysis["secondary_intents"] = [
                    {"intent": intent, "score": data["score"]}
                    for intent, data in sorted_intents[1:3]
                ]

            # 답변 유형 결정
            primary = best_intent[0]
            if "기관" in primary:
                intent_analysis["answer_type_required"] = "기관명"
                intent_analysis["context_hints"].append("구체적인 기관명 필요")
            elif "특징" in primary:
                intent_analysis["answer_type_required"] = "특징설명"
                intent_analysis["context_hints"].append("특징과 성질 나열")
            elif "지표" in primary:
                intent_analysis["answer_type_required"] = "지표나열"
                intent_analysis["context_hints"].append("탐지 지표와 징후")
            elif "방안" in primary:
                intent_analysis["answer_type_required"] = "방안제시"
                intent_analysis["context_hints"].append("구체적 실행방안")
            elif "절차" in primary:
                intent_analysis["answer_type_required"] = "절차설명"
                intent_analysis["context_hints"].append("단계별 절차")
            elif "조치" in primary:
                intent_analysis["answer_type_required"] = "조치설명"
                intent_analysis["context_hints"].append("보안조치 내용")

        # 추가 문맥 분석
        self._add_context_analysis(question, intent_analysis)

        return intent_analysis

    def _add_context_analysis(self, question: str, intent_analysis: Dict):
        """추가 문맥 분석"""
        question_lower = question.lower()

        # 긴급성 표시어 확인
        urgency_keywords = ["긴급", "즉시", "신속", "빠른"]
        if any(keyword in question_lower for keyword in urgency_keywords):
            intent_analysis["context_hints"].append("긴급 대응 필요")

        # 예시 요구 확인
        example_keywords = ["예시", "사례", "구체적", "실제"]
        if any(keyword in question_lower for keyword in example_keywords):
            intent_analysis["context_hints"].append("구체적 예시 포함")

        # 비교 요구 확인
        comparison_keywords = ["비교", "차이", "구별", "비교하여"]
        if any(keyword in question_lower for keyword in comparison_keywords):
            intent_analysis["context_hints"].append("비교 분석 필요")

        # 단계적 설명 요구 확인
        step_keywords = ["단계", "순서", "과정", "절차"]
        if any(keyword in question_lower for keyword in step_keywords):
            intent_analysis["context_hints"].append("단계별 설명 필요")

    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """선택지 범위 추출"""
        question_type = self.analyze_question_type(question)

        if question_type != "multiple_choice":
            return "subjective", 0

        # 줄바꿈으로 분리된 선택지 패턴 확인
        lines = question.split("\n")
        choice_numbers = []

        for line in lines:
            line = line.strip()
            match = re.match(r"^(\d+)\s+(.+)", line)
            if match:
                num = int(match.group(1))
                content = match.group(2).strip()
                if 1 <= num <= 5 and len(content) > 0:
                    choice_numbers.append(num)

        # 연속된 선택지인지 확인
        if choice_numbers:
            choice_numbers.sort()
            max_choice = max(choice_numbers)
            min_choice = min(choice_numbers)

            # 연속성 검증
            expected_count = max_choice - min_choice + 1
            if (
                len(choice_numbers) == expected_count
                and min_choice == 1
                and max_choice >= 3
            ):
                return "multiple_choice", max_choice

        # 전통적인 패턴으로 확인
        for i in range(5, 2, -1):
            pattern_parts = [f"{j}\\s+[가-힣\\w]+" for j in range(1, i + 1)]
            pattern = ".*".join(pattern_parts)
            if re.search(pattern, question, re.DOTALL):
                return "multiple_choice", i

        # 객관식 키워드가 있지만 선택지를 찾을 수 없는 경우
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                return "multiple_choice", 5

        return "subjective", 0

    def analyze_question_type(self, question: str) -> str:
        """질문 유형 분석"""

        question = question.strip()

        # 주관식 패턴 우선 확인
        for pattern in self.subj_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "subjective"

        # 실제 데이터 패턴 기반 객관식 확인
        choice_pattern = r"\n(\d+)\s+[가-힣\w]"
        choice_matches = re.findall(choice_pattern, question)

        if len(choice_matches) >= 3:
            # 선택지 번호가 연속적인지 확인
            choice_nums = [int(match) for match in choice_matches]
            choice_nums.sort()
            if (
                choice_nums[0] == 1
                and len(choice_nums) == choice_nums[-1]
                and choice_nums[-1] <= 5
            ):
                return "multiple_choice"

        # 객관식 키워드 확인
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                # 선택지가 있는지 추가 확인
                if any(f"{i} " in question for i in range(1, 6)):
                    return "multiple_choice"

        # 전통적인 객관식 패턴 확인
        for pattern in self.mc_patterns:
            if re.search(pattern, question, re.DOTALL | re.MULTILINE):
                return "multiple_choice"

        # 길이와 구조 기반 추정
        if (
            len(question) < 400
            and re.search(r"것은\?|것\?|것은\s*$", question)
            and len(re.findall(r"\b[1-5]\b", question)) >= 3
        ):
            return "multiple_choice"

        return "subjective"

    def extract_domain(self, question: str) -> str:
        """도메인 추출"""
        question_lower = question.lower()

        # 각 도메인별 키워드 매칭 점수 계산
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 핵심 키워드는 가중치 부여
                    if keyword in [
                        "개인정보보호법",
                        "전자금융거래법",
                        "자본시장법",
                        "ISMS",
                        "트로이",
                        "RAT",
                        "원격제어",
                        "분쟁조정",
                        "위험관리",
                    ]:
                        score += 3
                    else:
                        score += 1

            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return "일반"

        # 가장 높은 점수의 도메인 선택
        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]

        # 실제 데이터 분포에 맞는 추가 검증
        if detected_domain == "사이버보안":
            cybersec_keywords = [
                "트로이",
                "악성코드",
                "RAT",
                "원격제어",
                "딥페이크",
                "SBOM",
                "보안",
            ]
            if any(keyword in question_lower for keyword in cybersec_keywords):
                detected_domain = "사이버보안"
        elif detected_domain == "개인정보보호":
            privacy_keywords = ["개인정보", "정보주체", "만 14세", "법정대리인", "PIMS"]
            if any(keyword in question_lower for keyword in privacy_keywords):
                detected_domain = "개인정보보호"

        return detected_domain

    def clean_korean_text(self, text: str) -> str:
        """한국어 전용 텍스트 정리 - 완화된 처리"""
        if not text:
            return ""

        # 기본 복구
        text = self.restore_korean_characters(text)

        # 텍스트 품질 향상
        text = self.enhance_korean_text_quality(text)

        # 문법 구조 개선
        text = self.fix_grammatical_structure(text)

        # 기본 정리
        text = re.sub(r"\s+", " ", text).strip()

        # 깨진 문자 및 인코딩 오류 처리 (완화)
        text = re.sub(r"[^\w\s가-힣.,!?()[\]\-]", " ", text)

        # 영어 문자 제거 (완화된 기준)
        english_chars = len(re.findall(r"[a-zA-Z]", text))
        total_chars = len(re.sub(r"[^\w가-힣]", "", text))
        if total_chars > 0 and english_chars / total_chars > 0.5:  # 50% 이상일 때만 제거
            text = re.sub(r"[a-zA-Z]+", "", text)

        # 중국어 제거
        text = re.sub(r"[\u4e00-\u9fff]", "", text)

        # 특수 기호 제거
        text = re.sub(r"[①②③④⑤➀➁➂➃➄]", "", text)

        # 반복 공백 제거
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0

        korean_chars = len(re.findall(r"[가-힣]", text))
        total_chars = len(re.sub(r"[^\w가-힣]", "", text))

        if total_chars == 0:
            return 0.0

        return korean_chars / total_chars

    def calculate_english_ratio(self, text: str) -> float:
        """영어 비율 계산"""
        if not text:
            return 0.0

        english_chars = len(re.findall(r"[a-zA-Z]", text))
        total_chars = len(re.sub(r"[^\w가-힣]", "", text))

        if total_chars == 0:
            return 0.0

        return english_chars / total_chars

    def validate_mc_answer_range(self, answer: str, max_choice: int) -> bool:
        """객관식 답변 범위 검증"""
        if not answer or not answer.isdigit():
            return False

        answer_num = int(answer)
        return 1 <= answer_num <= max_choice

    def validate_answer_intent_match(
        self, answer: str, question: str, intent_analysis: Dict
    ) -> bool:
        """답변과 질문 의도 일치성 검증 - 완화"""
        if not answer or not intent_analysis:
            return True  # 기본적으로 통과

        # 치명적인 반복 패턴이 있으면 즉시 실패
        if self.detect_critical_repetitive_patterns(answer):
            return False

        required_type = intent_analysis.get("answer_type_required", "설명형")
        answer_lower = answer.lower()

        # 기관명이 필요한 경우만 엄격하게 검증
        if required_type == "기관명":
            institution_keywords = [
                "위원회",
                "감독원",
                "은행",
                "기관",
                "센터",
                "청",
                "부",
                "원",
            ]

            keyword_count = sum(
                1 for keyword in institution_keywords if keyword in answer_lower
            )

            return keyword_count >= 1

        # 나머지는 기본적으로 통과
        return True

    def validate_korean_answer(
        self, answer: str, question_type: str, max_choice: int = 5, question: str = ""
    ) -> bool:
        """한국어 답변 유효성 검증 - 대폭 완화"""
        if not answer:
            return False

        answer = str(answer).strip()

        # 치명적인 반복 패턴만 조기 감지
        if self.detect_critical_repetitive_patterns(answer):
            return False

        if question_type == "multiple_choice":
            # 객관식: 지정된 범위의 숫자
            if not self.validate_mc_answer_range(answer, max_choice):
                return False
            return True

        else:
            # 주관식: 매우 완화된 검증
            clean_answer = self.clean_korean_text(answer)

            # 길이 검증 (완화)
            if len(clean_answer) < 5:  # 15에서 5로 완화
                return False

            # 한국어 비율 검증 (완화)
            korean_ratio = self.calculate_korean_ratio(clean_answer)
            if korean_ratio < 0.2:  # 매우 완화
                return False

            # 폴백 메시지 확인
            if "생성에 실패" in clean_answer:
                return False

            return True

    def validate_answer(
        self, answer: str, question_type: str, max_choice: int = 5, question: str = ""
    ) -> bool:
        """답변 유효성 검증"""
        return self.validate_korean_answer(answer, question_type, max_choice, question)

    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        return self.clean_korean_text(text)

    def extract_choices(self, question: str) -> List[str]:
        """객관식 선택지 추출"""
        choices = []

        # 실제 데이터 패턴: "1 소비자금융업\n2 투자자문업\n3 투자매매업"
        lines = question.split("\n")
        for line in lines:
            line = line.strip()
            match = re.match(r"^(\d+)\s+(.+)", line)
            if match:
                choice_num = int(match.group(1))
                choice_content = match.group(2).strip()
                if 1 <= choice_num <= 5 and len(choice_content) > 0:
                    choices.append(choice_content)

        # 순서 정렬
        if len(choices) >= 3:
            return choices

        # 폴백: 전통적인 패턴으로도 확인
        if not choices:
            patterns = [
                r"(\d+)\s+([^0-9\n]+?)(?=\d+\s+|$)",
                r"(\d+)\)\s*([^0-9\n]+?)(?=\d+\)|$)",
                r"(\d+)\.\s*([^0-9\n]+?)(?=\d+\.|$)",
                r"[①②③④⑤]\s*([^①②③④⑤\n]+?)(?=[①②③④⑤]|$)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, question, re.MULTILINE | re.DOTALL)
                if matches:
                    if isinstance(matches[0], tuple):
                        choices = [match[1].strip() for match in matches]
                    else:
                        choices = [match.strip() for match in matches]

                    if len(choices) >= 3:
                        break

        return choices[:5]

    def analyze_question_difficulty(self, question: str) -> str:
        """질문 난이도 분석"""
        question_lower = question.lower()

        # 전문 용어 개수
        technical_terms = [
            "isms",
            "pims",
            "sbom",
            "원격제어",
            "침입탐지",
            "트로이",
            "멀웨어",
            "랜섬웨어",
            "딥페이크",
            "피싱",
            "접근매체",
            "전자서명",
            "개인정보보호법",
            "자본시장법",
            "rat",
            "원격접근",
            "탐지지표",
            "apt",
            "ddos",
            "ids",
            "ips",
            "bcp",
            "drp",
            "isms-p",
            "분쟁조정",
            "금융투자업",
            "위험관리",
            "재해복구",
            "비상연락체계",
        ]

        term_count = sum(1 for term in technical_terms if term in question_lower)

        # 문장 길이
        length = len(question)

        # 선택지 개수
        choice_count = len(self.extract_choices(question))

        # 난이도 계산
        if term_count >= 3 or length > 400 or choice_count >= 5:
            return "고급"
        elif term_count >= 1 or length > 200 or choice_count >= 4:
            return "중급"
        else:
            return "초급"

    def normalize_korean_answer(
        self, answer: str, question_type: str, max_choice: int = 5
    ) -> str:
        """한국어 답변 정규화 - 완화된 처리"""
        if not answer:
            return ""

        answer = str(answer).strip()

        if question_type == "multiple_choice":
            # 숫자만 추출하고 범위 검증
            numbers = re.findall(r"[1-9]", answer)
            for num in numbers:
                if 1 <= int(num) <= max_choice:
                    return num

            return ""

        else:
            # 주관식 답변 한국어 정리
            answer = self.clean_korean_text(answer)

            # 치명적인 반복 패턴만 최종 확인
            if self.detect_critical_repetitive_patterns(answer):
                return "생성에 실패하였습니다."

            # 의미 없는 짧은 문장 제거 (완화)
            if len(answer) < 5:  # 15에서 5로 완화
                return "생성에 실패하였습니다."

            # 길이 제한 (완화)
            if len(answer) > self.korean_requirements["max_length"]:
                sentences = answer.split(". ")
                valid_sentences = []

                for sentence in sentences:
                    if not self.detect_critical_repetitive_patterns(sentence):
                        valid_sentences.append(sentence)
                    if len(valid_sentences) >= 6:  # 4에서 6으로 증가
                        break

                if valid_sentences:
                    answer = ". ".join(valid_sentences[:6])
                else:
                    return "생성에 실패하였습니다."

                if len(answer) > self.korean_requirements["max_length"]:
                    answer = answer[: self.korean_requirements["max_length"]]

            # 마침표 확인
            if answer and not answer.endswith((".", "다", "요", "함")):
                answer += "."

            return answer

    def normalize_answer(
        self, answer: str, question_type: str, max_choice: int = 5
    ) -> str:
        """답변 정규화"""
        return self.normalize_korean_answer(answer, question_type, max_choice)

    def cleanup(self):
        """정리"""
        pass