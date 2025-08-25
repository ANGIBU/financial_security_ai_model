# data_processor.py

import re
import unicodedata
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

from config import KOREAN_REQUIREMENTS


class SimpleDataProcessor:

    def __init__(self):
        self._initialize_integrated_data()
        self.korean_requirements = KOREAN_REQUIREMENTS.copy()
        # 검증 기준 조정
        self.korean_requirements["min_korean_ratio"] = 0.4
        self.korean_requirements["max_english_ratio"] = 0.5
        self.korean_requirements["min_length"] = 15

    def _initialize_integrated_data(self):
        """JSON 데이터를 코드 내부로 통합하여 초기화"""
        
        # 객관식 패턴 - 더 정확한 패턴 추가
        self.mc_patterns = [
            r"1\s+[가-힣\w].*\n2\s+[가-힣\w].*\n3\s+[가-힣\w]",
            r"①.*②.*③.*④.*⑤",
            r"1\s+[가-힣].*2\s+[가-힣].*3\s+[가-힣].*4\s+[가-힣].*5\s+[가-힣]",
            r"1\s+.*2\s+.*3\s+.*4\s+.*5\s+",
            r"1\s+.*2\s+.*3\s+.*4\s+",
            r"1\.\s*.*2\.\s*.*3\.\s*.*4\.\s*.*5\.",
            r"1\.\s*.*2\.\s*.*3\.\s*.*4\.",
            r"1\)\s*.*2\)\s*.*3\)\s*.*4\)\s*.*5\)",
            r"1\)\s*.*2\)\s*.*3\)\s*.*4\)"
        ]

        # 객관식 키워드
        self.mc_keywords = [
            r"해당하지.*않는.*것",
            r"적절하지.*않는.*것", 
            r"옳지.*않는.*것",
            r"틀린.*것",
            r"맞는.*것",
            r"옳은.*것",
            r"적절한.*것",
            r"올바른.*것",
            r"가장.*적절한.*것",
            r"가장.*옳은.*것",
            r"구분.*해당하지.*않는.*것",
            r"다음.*중.*것은",
            r"다음.*중.*것",
            r"다음.*보기.*중",
            r"무엇인가\?$",
            r"어떤.*것인가\?$",
            r"몇.*개인가\?$"
        ]

        # 질문 의도 패턴
        self.question_intent_patterns = {
            "기관_묻기": [
                r"기관.*기술하세요",
                r"기관.*설명하세요", 
                r"기관.*서술하세요",
                r"기관.*무엇",
                r"어떤.*기관",
                r"어느.*기관",
                r"기관.*어디",
                r"분쟁조정.*신청.*기관",
                r"조정.*신청.*기관",
                r"분쟁.*조정.*기관", 
                r"신청.*수.*있는.*기관",
                r"분쟁.*해결.*기관",
                r"조정.*담당.*기관",
                r"감독.*기관",
                r"관리.*기관",
                r"담당.*기관",
                r"주관.*기관",
                r"소관.*기관",
                r"신고.*기관",
                r"접수.*기관",
                r"상담.*기관",
                r"문의.*기관",
                r"위원회.*무엇",
                r"위원회.*어디",
                r"위원회.*설명",
                r"전자금융.*분쟁.*기관",
                r"전자금융.*조정.*기관",
                r"개인정보.*신고.*기관",
                r"개인정보.*보호.*기관",
                r"개인정보.*침해.*기관",
                r"기관을.*기술하세요",
                r".*기관.*기술",
                r"분쟁조정.*기관",
                r"신청할.*수.*있는.*기관"
            ],
            "특징_묻기": [
                r"특징.*설명하세요",
                r"특징.*기술하세요",
                r"특징.*서술하세요", 
                r"어떤.*특징",
                r"주요.*특징",
                r"특징.*무엇",
                r"성격.*설명",
                r"성질.*설명",
                r"속성.*설명",
                r"특성.*설명",
                r"특성.*무엇",
                r"성격.*무엇",
                r"특성.*기술",
                r"속성.*기술",
                r"기반.*원격제어.*악성코드.*특징",
                r"트로이.*특징",
                r"RAT.*특징",
                r".*특징.*설명하세요",
                r".*특징.*기술하세요",
                r"트로이.*목마.*특징",
                r"원격제어.*악성코드.*특징",
                r"RAT.*특징",
                r"악성코드.*특징"
            ],
            "지표_묻기": [
                r"지표.*설명하세요",
                r"탐지.*지표",
                r"주요.*지표", 
                r"어떤.*지표",
                r"지표.*무엇",
                r"징후.*설명",
                r"신호.*설명",
                r"패턴.*설명",
                r"행동.*패턴",
                r"활동.*패턴",
                r"모니터링.*지표",
                r"관찰.*지표",
                r"식별.*지표",
                r"발견.*방법",
                r"탐지.*방법",
                r"주요.*탐지.*지표",
                r"악성코드.*탐지.*지표",
                r"원격제어.*탐지.*지표",
                r".*탐지.*지표.*설명하세요",
                r".*지표.*설명하세요",
                r"주요.*탐지.*지표",
                r"탐지.*지표.*무엇"
            ],
            "방안_묻기": [
                r"방안.*기술하세요",
                r"방안.*설명하세요",
                r"대응.*방안",
                r"해결.*방안",
                r"관리.*방안",
                r"어떤.*방안",
                r"대책.*설명", 
                r"조치.*방안",
                r"처리.*방안",
                r"개선.*방안",
                r"예방.*방안",
                r"보완.*방안",
                r"강화.*방안",
                r"딥페이크.*대응.*방안",
                r"금융권.*대응.*방안",
                r"악용.*대비.*방안"
            ],
            "절차_묻기": [
                r"절차.*설명하세요",
                r"절차.*기술하세요",
                r"어떤.*절차",
                r"처리.*절차",
                r"진행.*절차",
                r"수행.*절차",
                r"실행.*절차",
                r"과정.*설명",
                r"단계.*설명", 
                r"프로세스.*설명"
            ],
            "조치_묻기": [
                r"조치.*설명하세요",
                r"조치.*기술하세요",
                r"어떤.*조치",
                r"보안.*조치",
                r"대응.*조치",
                r"예방.*조치",
                r"보완.*조치"
            ]
        }

        # 주관식 패턴
        self.subj_patterns = [
            r"설명하세요",
            r"기술하세요",
            r"서술하세요",
            r"작성하세요",
            r"무엇인가요",
            r"어떻게.*해야.*하며",
            r"방안을.*기술",
            r"대응.*방안",
            r"특징.*다음과.*같",
            r"탐지.*지표",
            r"행동.*패턴",
            r"분석하여.*제시",
            r"조치.*사항",
            r"제시하시오",
            r"논하시오",
            r"답하시오",
            r"특징과.*주요.*탐지.*지표를.*설명하세요",
            r"기관을.*기술하세요",
            r"대응.*방안을.*기술하세요"
        ]

        # 도메인 키워드
        self.domain_keywords = {
            "개인정보보호": {
                "primary": ["개인정보보호법", "개인정보보호위원회", "개인정보침해신고센터"],
                "secondary": ["개인정보", "정보주체", "민감정보", "고유식별정보", "만 14세", "법정대리인", "PIMS"],
                "general": ["수집", "이용", "제공", "파기", "동의", "처리", "열람권", "정정삭제권"]
            },
            "전자금융": {
                "primary": ["전자금융거래법", "전자금융분쟁조정위원회", "금융감독원"],
                "secondary": ["전자금융", "전자적", "접근매체", "분쟁조정", "한국은행"],
                "general": ["전자서명", "전자인증", "이용자", "금융통화위원회", "지급결제제도"]
            },
            "사이버보안": {
                "primary": ["트로이", "RAT", "원격제어 악성코드", "SBOM"],
                "secondary": ["악성코드", "딥페이크", "원격제어", "소프트웨어 구성 요소"],
                "general": ["멀웨어", "바이러스", "피싱", "랜섬웨어", "해킹", "백도어", "탐지"]
            },
            "정보보안": {
                "primary": ["정보보안관리체계", "ISMS"],
                "secondary": ["정보보안", "보안관리", "보안정책", "접근통제"],
                "general": ["암호화", "방화벽", "침입탐지", "IDS", "IPS", "보안관제"]
            },
            "금융투자": {
                "primary": ["자본시장법", "금융투자업"],
                "secondary": ["투자자문업", "투자매매업", "투자중개업", "소비자금융업", "보험중개업"],
                "general": ["집합투자업", "신탁업", "투자자보호", "적합성원칙"]
            },
            "위험관리": {
                "primary": ["위험관리"],
                "secondary": ["위험평가", "위험대응", "위험수용", "내부통제"],
                "general": ["컴플라이언스", "위험식별", "위험분석", "위험모니터링"]
            }
        }

        self._setup_korean_recovery_mappings()

    def _setup_korean_recovery_mappings(self):
        """한국어 복구 매핑 설정"""
        self.korean_recovery_mapping = {
            "갈취 묻는 말": "",
            "묻고 갈취": "", 
            "갈취": "",
            "() 기반": "",
            "()는": "",
            "() 특징": "",
            "() 지표": "",
            "() 방안": "",
            "()를": "",
            "()에": "",
            "()의": "",
            "()와": "",
            "()로": "",
            "괄호": "",
            "(괄호)": "",
        }

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

        # 의도별 점수 계산
        intent_scores = {}

        for intent_type, patterns in self.question_intent_patterns.items():
            score = 0
            matched_patterns = []

            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    # 패턴 매칭 정확도에 따른 가중치
                    if len(re.findall(pattern, question, re.IGNORECASE)) > 1:
                        score += 4.0  # 여러 번 매칭
                    else:
                        score += 3.0  # 한 번 매칭
                    matched_patterns.append(pattern)

            # 복합 질문 보너스 (특징+지표)
            if intent_type == "특징_묻기" and "지표" in question_lower:
                score += 5.0
                intent_analysis["answer_type_required"] = "복합설명"
            elif intent_type == "지표_묻기" and "특징" in question_lower:
                score += 5.0
                intent_analysis["answer_type_required"] = "복합설명"

            # 정확한 키워드 매칭 보너스
            exact_matches = {
                "기관_묻기": ["기관을 기술하세요", "분쟁조정을 신청할 수 있는 기관"],
                "특징_묻기": ["특징을 설명하세요", "트로이 목마의 특징"],
                "지표_묻기": ["탐지 지표를 설명하세요", "주요 탐지 지표"],
                "방안_묻기": ["대응 방안을 기술하세요", "방안을 기술하세요"]
            }

            if intent_type in exact_matches:
                for exact_phrase in exact_matches[intent_type]:
                    if exact_phrase in question_lower:
                        score += 6.0

            if score > 0:
                intent_scores[intent_type] = {
                    "score": score,
                    "patterns": matched_patterns,
                }

        # 최고 점수 의도 선택
        if intent_scores:
            sorted_intents = sorted(
                intent_scores.items(), key=lambda x: x[1]["score"], reverse=True
            )
            best_intent = sorted_intents[0]

            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = min(best_intent[1]["score"] / 8.0, 1.0)
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]

            if len(sorted_intents) > 1:
                intent_analysis["secondary_intents"] = [
                    {"intent": intent, "score": data["score"]}
                    for intent, data in sorted_intents[1:3]
                ]

            # 답변 유형 결정
            primary = best_intent[0]
            if "기관" in primary:
                intent_analysis["answer_type_required"] = "기관명"
                intent_analysis["context_hints"].append("구체적인 기관명과 업무")
            elif "특징" in primary:
                intent_analysis["answer_type_required"] = "특징설명"
                intent_analysis["context_hints"].append("주요 특징과 기술적 특성")
            elif "지표" in primary:
                intent_analysis["answer_type_required"] = "지표나열"
                intent_analysis["context_hints"].append("탐지 지표와 모니터링 방법")
            elif "방안" in primary:
                intent_analysis["answer_type_required"] = "방안제시"
                intent_analysis["context_hints"].append("구체적 실행 방안")

        # 복합 질문 특별 처리
        if ("특징" in question_lower and "지표" in question_lower):
            intent_analysis["primary_intent"] = "복합설명"
            intent_analysis["answer_type_required"] = "복합설명"
            intent_analysis["context_hints"].append("특징과 지표 모두 포함")

        return intent_analysis

    def extract_domain(self, question: str) -> str:
        """도메인 추출"""
        question_lower = question.lower()

        domain_scores = {}

        for domain, keyword_groups in self.domain_keywords.items():
            score = 0
            
            # 우선순위별 키워드 점수 계산
            for primary_keyword in keyword_groups["primary"]:
                if primary_keyword.lower() in question_lower:
                    score += 10  # 높은 점수

            for secondary_keyword in keyword_groups["secondary"]:
                if secondary_keyword.lower() in question_lower:
                    score += 5   # 중간 점수

            for general_keyword in keyword_groups["general"]:
                if general_keyword.lower() in question_lower:
                    score += 1   # 낮은 점수

            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return "일반"

        # 최고 점수 도메인 선택
        best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]

        # 추가 검증
        if best_domain == "사이버보안":
            if any(term in question_lower for term in ["트로이", "rat", "원격제어", "악성코드", "딥페이크", "sbom"]):
                return "사이버보안"
        elif best_domain == "전자금융":
            if any(term in question_lower for term in ["전자금융", "분쟁조정", "금융감독원", "한국은행"]):
                return "전자금융"
        elif best_domain == "개인정보보호":
            if any(term in question_lower for term in ["개인정보", "정보주체", "만 14세", "개인정보보호위원회"]):
                return "개인정보보호"

        return best_domain

    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """선택지 범위 추출"""
        question_type = self.analyze_question_type(question)

        if question_type != "multiple_choice":
            return "subjective", 0

        # 정확한 선택지 번호 추출
        lines = question.split("\n")
        choice_numbers = []

        for line in lines:
            line = line.strip()
            # 다양한 선택지 패턴 매칭
            patterns = [
                r"^(\d+)\s+(.+)",        # "1 내용"
                r"^(\d+)\.\s*(.+)",      # "1. 내용"  
                r"^(\d+)\)\s*(.+)",      # "1) 내용"
                r"^[①②③④⑤]\s*(.+)"      # "① 내용"
            ]
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    if pattern.startswith(r"^[①②③④⑤]"):
                        # 원형 숫자를 일반 숫자로 변환
                        circle_nums = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
                        for circle, num in circle_nums.items():
                            if line.startswith(circle):
                                choice_numbers.append(num)
                                break
                    else:
                        num = int(match.group(1))
                        content = match.group(2).strip()
                        if 1 <= num <= 5 and len(content) > 2:
                            choice_numbers.append(num)
                    break

        if choice_numbers:
            choice_numbers = sorted(list(set(choice_numbers)))
            max_choice = max(choice_numbers)
            min_choice = min(choice_numbers)

            # 연속된 선택지인지 확인
            expected_count = max_choice - min_choice + 1
            if (len(choice_numbers) == expected_count and 
                min_choice == 1 and 
                3 <= max_choice <= 5):
                return "multiple_choice", max_choice

        # 패턴 기반 추정
        for i in range(5, 2, -1):
            pattern_parts = [f"{j}\\s+[가-힣\\w]+" for j in range(1, i + 1)]
            pattern = ".*".join(pattern_parts)
            if re.search(pattern, question, re.DOTALL):
                return "multiple_choice", i

        # 키워드 기반 추정
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                # 실제 선택지 개수 확인
                number_count = len(re.findall(r'\b[1-5]\b', question))
                if number_count >= 3:
                    # 4개/5개 구분을 위해 실제 선택지 확인
                    actual_choices = []
                    for line in question.split('\n'):
                        match = re.match(r'^(\d+)\s+', line.strip())
                        if match:
                            actual_choices.append(int(match.group(1)))
                    if actual_choices:
                        return "multiple_choice", max(actual_choices)
                    return "multiple_choice", 5

        return "subjective", 0

    def analyze_question_type(self, question: str) -> str:
        """질문 유형 분석"""
        question = question.strip()

        # 1단계: 주관식 패턴 우선 확인
        for pattern in self.subj_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "subjective"

        # 2단계: 객관식 선택지 패턴 확인
        lines = question.split("\n")
        choice_lines = 0
        
        for line in lines:
            line = line.strip()
            # 선택지 라인 패턴 매칭
            if (re.match(r"^[1-5]\s+", line) or 
                re.match(r"^[1-5]\.\s*", line) or
                re.match(r"^[1-5]\)\s*", line) or
                re.match(r"^[①②③④⑤]\s*", line)):
                if len(line) > 5:  # 의미있는 내용이 있는 경우
                    choice_lines += 1

        if choice_lines >= 3:
            return "multiple_choice"

        # 3단계: 객관식 키워드 확인
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                # 선택지 번호가 있는지 추가 확인
                if len(re.findall(r'\b[1-5]\b', question)) >= 3:
                    return "multiple_choice"

        # 4단계: 객관식 패턴 확인
        for pattern in self.mc_patterns:
            if re.search(pattern, question, re.DOTALL | re.MULTILINE):
                return "multiple_choice"

        # 5단계: 마지막 휴리스틱
        if (len(question) < 500 and 
            re.search(r"것은\?|것\?|것은\s*$", question) and
            len(re.findall(r"\b[1-5]\b", question)) >= 3):
            return "multiple_choice"

        return "subjective"

    def validate_answer(self, answer: str, question_type: str, 
                       max_choice: int = 5, question: str = "") -> bool:
        """답변 검증"""
        if not answer:
            return False

        answer = str(answer).strip()

        # 반복 패턴 체크
        if self.detect_critical_repetitive_patterns(answer):
            return False

        if question_type == "multiple_choice":
            return self.validate_mc_answer_range(answer, max_choice)
        else:
            return self.validate_subjective_answer(answer, question)

    def validate_subjective_answer(self, answer: str, question: str = "") -> bool:
        """주관식 답변 검증"""
        
        clean_answer = self.clean_korean_text(answer)
        
        if self.detect_critical_repetitive_patterns(clean_answer):
            return False

        # 길이 검증
        if len(clean_answer) < 15:
            return False

        # 한국어 비율 검증
        korean_ratio = self.calculate_korean_ratio(clean_answer)
        if korean_ratio < 0.4:
            return False

        # 영어 비율 검증
        english_ratio = self.calculate_english_ratio(answer)
        if english_ratio > 0.5:
            return False

        # 한국어 문자 최소 개수 검증
        korean_chars = len(re.findall(r"[가-힣]", clean_answer))
        if korean_chars < 5:
            return False

        # 의미있는 키워드 검증
        meaningful_keywords = [
            "법", "규정", "조치", "관리", "보안", "방안", "절차", "기준",
            "정책", "체계", "시스템", "통제", "특징", "지표", "탐지", "대응",
            "기관", "위원회", "감독원", "업무", "담당", "수행", "필요", "해야",
            "구축", "수립", "시행", "실시", "트로이", "악성코드", "원격제어",
            "전자금융", "분쟁조정", "개인정보", "네트워크", "모니터링", "분석",
            "있", "는", "다", "을", "를", "의", "에", "와", "과", "로", "으로",
            "안전", "위험", "보호", "운영", "활동", "처리", "정보", "데이터",
            "서비스", "사용", "이용", "제공", "확인", "검토", "점검", "감사", "교육"
        ]

        if any(word in clean_answer for word in meaningful_keywords):
            return True

        # 길이가 충분하면 통과
        if len(clean_answer) >= 30:
            return True

        return False

    def validate_mc_answer_range(self, answer: str, max_choice: int) -> bool:
        """객관식 답변 범위 확인"""
        if not answer or not str(answer).strip().isdigit():
            return False

        answer_num = int(str(answer).strip())
        return 1 <= answer_num <= max_choice

    def detect_critical_repetitive_patterns(self, text: str) -> bool:
        """문제 패턴 감지"""
        if not text or len(text) < 30:
            return False

        critical_patterns = [
            r"갈취 묻는 말",
            r"묻고 갈취", 
            r"(.{1,3})\s*(\1\s*){12,}",
        ]

        for pattern in critical_patterns:
            if re.search(pattern, text):
                return True

        words = text.split()
        if len(words) >= 12:
            for i in range(len(words) - 11):
                same_count = 1
                for j in range(i + 1, min(i + 12, len(words))):
                    if words[i] == words[j]:
                        same_count += 1
                    else:
                        break

                if same_count >= 12 and len(words[i]) <= 5:
                    return True

        return False

    def clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리"""
        if not text:
            return ""

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)
            if len(text) < 8:
                return "텍스트 정리 중 내용이 부족합니다."

        text = self.restore_korean_characters(text)
        text = self.fix_grammatical_structure(text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\w\s가-힣.,!?()[\]\-]", " ", text)

        # 영어 비율 체크
        english_chars = len(re.findall(r"[a-zA-Z]", text))
        total_chars = len(re.sub(r"[^\w가-힣]", "", text))
        if total_chars > 0 and english_chars / total_chars > 0.5:
            text = re.sub(r"[a-zA-Z]+", "", text)

        # 중국어, 일본어 문자 제거
        text = re.sub(r"[\u4e00-\u9fff]", "", text)
        text = re.sub(r"[①②③④⑤➀➁➂➃➄]", "", text)

        text = re.sub(r"\s+", " ", text).strip()

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)
            if len(text) < 10:
                return "텍스트 정리 후 내용이 부족합니다."

        return text

    def remove_critical_repetitive_patterns(self, text: str) -> str:
        """문제 패턴 제거"""
        if not text:
            return ""

        for pattern, replacement in self.korean_recovery_mapping.items():
            text = text.replace(pattern, replacement)

        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            current_word = words[i]
            count = 1
            while i + count < len(words) and words[i + count] == current_word:
                count += 1

            if len(current_word) <= 2:
                cleaned_words.extend([current_word] * min(3, count))
            elif len(current_word) <= 5:
                cleaned_words.extend([current_word] * min(5, count))
            elif count >= 10:
                cleaned_words.extend([current_word] * min(5, count))
            else:
                cleaned_words.extend([current_word] * count)

            i += count

        text = " ".join(cleaned_words)
        text = re.sub(r"(.{3,15})\s*\1\s*\1\s*\1\s*\1\s*\1+", r"\1", text)
        text = re.sub(r"(.{1,5})\s*(\1\s*){10,}", r"\1", text)
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\s*\(\s*\)\s*", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def restore_korean_characters(self, text: str) -> str:
        """한국어 문자 복구"""
        if not text:
            return ""

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)

        text = unicodedata.normalize("NFC", text)

        for broken, correct in self.korean_recovery_mapping.items():
            text = text.replace(broken, correct)

        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"[.,!?]{3,}", ".", text)
        text = re.sub(r"\s+[.,!?]\s+", ". ", text)

        return text

    def fix_grammatical_structure(self, text: str) -> str:
        """문법 구조 수정"""
        if not text:
            return ""

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)

        # 기본적인 문법 수정
        grammar_fixes = [
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
            (r"([가-힣])\s+다\s*\.", r"\1다."),
            (r"([가-힣])\s+요\s*\.", r"\1요."),
            (r"([가-힣])\s+함\s*\.", r"\1함."),
            (r"([가-힣])\s+니다\s*\.", r"\1니다."),
            (r"([가-힣])\s+습니다\s*\.", r"\1습니다."),
            (r"\.+", "."),
            (r"\s*\.\s*", ". "),
            (r"\s*,\s*", ", "),
            (r"([가-힣])\s*$", r"\1."),
        ]

        for pattern, replacement in grammar_fixes:
            text = re.sub(pattern, replacement, text)

        # 문장별 처리
        sentences = text.split(".")
        processed_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            if self.detect_critical_repetitive_patterns(sentence):
                continue

            if len(sentence) > 250:
                parts = re.split(r"[,，]", sentence)
                if len(parts) > 1:
                    for part in parts:
                        part = part.strip()
                        if len(part) > 8 and not self.detect_critical_repetitive_patterns(part):
                            processed_sentences.append(part)
                else:
                    if not self.detect_critical_repetitive_patterns(sentence):
                        processed_sentences.append(sentence)
            else:
                if not self.detect_critical_repetitive_patterns(sentence):
                    processed_sentences.append(sentence)

        if processed_sentences:
            result = ". ".join(processed_sentences)
        else:
            result = "관련 법령과 규정에 따라 체계적인 관리를 수행해야 합니다"

        if result and not result.endswith("."):
            result += "."

        return result

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

    def analyze_question_difficulty(self, question: str) -> str:
        """질문 난이도 분석"""
        question_lower = question.lower()

        technical_terms = [
            "isms", "pims", "sbom", "원격제어", "침입탐지", "트로이", "멀웨어",
            "랜섬웨어", "딥페이크", "피싱", "접근매체", "전자서명",
            "개인정보보호법", "자본시장법", "rat", "원격접근", "탐지지표",
            "apt", "ddos", "ids", "ips", "bcp", "drp", "isms-p",
            "분쟁조정", "금융투자업", "위험관리", "재해복구", "비상연락체계",
        ]

        term_count = sum(1 for term in technical_terms if term in question_lower)
        length = len(question)
        
        if term_count >= 3 or length > 400:
            return "고급"
        elif term_count >= 2 or length > 200:
            return "중급"
        else:
            return "초급"

    def normalize_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """답변 정규화"""
        if not answer:
            return ""

        answer = str(answer).strip()

        if question_type == "multiple_choice":
            numbers = re.findall(r"[1-9]", answer)
            for num in numbers:
                if 1 <= int(num) <= max_choice:
                    return num
            return ""
        else:
            answer = self.clean_korean_text(answer)

            if self.detect_critical_repetitive_patterns(answer):
                if len(answer) > 50:
                    answer = self.remove_critical_repetitive_patterns(answer)
                    if len(answer) < 15:
                        return "답변 생성 중 반복 패턴이 감지되어 재생성이 필요합니다."
                else:
                    return "답변 생성 중 반복 패턴이 감지되어 재생성이 필요합니다."

            if len(answer) < 15:
                return "답변 길이가 부족하여 생성에 실패했습니다."

            # 길이 조정
            if len(answer) > 700:
                sentences = answer.split(". ")
                valid_sentences = []

                for sentence in sentences:
                    if not self.detect_critical_repetitive_patterns(sentence):
                        valid_sentences.append(sentence)
                    if len(valid_sentences) >= 5:
                        break

                if valid_sentences:
                    answer = ". ".join(valid_sentences[:5])
                else:
                    return "답변 정규화 중 유효한 문장을 찾을 수 없습니다."

                if len(answer) > 700:
                    answer = answer[:700]

            # 문장 끝 처리
            if answer and not answer.endswith((".", "다", "요", "함")):
                answer += "."

            return answer

    def cleanup(self):
        """리소스 정리"""
        pass