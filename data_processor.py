# data_processor.py

"""
데이터 처리기 - 의도 분석 및 답변 검증 대폭 강화
- 정교한 질문 의도 분석
- 컨텍스트 기반 답변 검증
- 자연스러운 한국어 처리
- 스마트 품질 평가
- 의도-답변 일치성 검증 강화
"""

import re
import json
import unicodedata
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

# 설정 파일 import
from config import KOREAN_REQUIREMENTS, JSON_CONFIG_FILES


class SimpleDataProcessor:
    """데이터 처리기 - 의도 분석 및 답변 검증 특화"""

    def __init__(self):
        # JSON 설정 파일 로드
        self._load_json_configs()

        # 한국어 전용 검증 기준 - 스마트 조정
        self.korean_requirements = KOREAN_REQUIREMENTS.copy()
        self.korean_requirements["min_korean_ratio"] = 0.6    # 균형잡힌 기준
        self.korean_requirements["max_english_ratio"] = 0.2   # 더 엄격한 기준
        self.korean_requirements["min_length"] = 20           # 적절한 기준

        # 의도 분석 강화 설정
        self.intent_analysis_config = {
            "confidence_threshold": 0.3,      # 낮춰서 더 포괄적으로
            "context_weight": 0.4,           # 컨텍스트 가중치
            "keyword_weight": 0.3,           # 키워드 가중치  
            "pattern_weight": 0.3,           # 패턴 가중치
            "semantic_analysis": True,       # 의미 분석 활성화
            "multi_intent_detection": True,  # 다중 의도 감지
        }

        # 답변 품질 평가 강화
        self.answer_quality_config = {
            "structure_weight": 0.25,        # 구조 가중치
            "content_weight": 0.35,          # 내용 가중치
            "language_weight": 0.25,         # 언어 품질 가중치
            "intent_match_weight": 0.15,     # 의도 일치 가중치
            "professional_level": True,      # 전문성 평가
            "natural_flow": True,            # 자연스러운 흐름
        }

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

            # 향상된 의도 분석 패턴 설정
            self._setup_enhanced_intent_patterns()

            print("데이터 처리 설정 파일 로드 완료 - 의도 분석 강화")

        except FileNotFoundError as e:
            print(f"설정 파일을 찾을 수 없습니다: {e}")
            self._load_default_configs()
        except json.JSONDecodeError as e:
            print(f"JSON 파일 파싱 오류: {e}")
            self._load_default_configs()
        except Exception as e:
            print(f"설정 파일 로드 중 오류: {e}")
            self._load_default_configs()

    def _setup_enhanced_intent_patterns(self):
        """향상된 의도 분석 패턴 설정"""
        
        # 확장된 의도 패턴
        self.enhanced_intent_patterns = {
            "기관_묻기": {
                "direct_patterns": [
                    r"어떤\s*기관", r"어느\s*기관", r"무엇.*기관", r"기관.*무엇",
                    r"담당.*기관", r"관리.*기관", r"감독.*기관", r"소관.*기관",
                    r"기관.*담당", r"기관.*관리", r"기관.*감독", r"기관.*소관",
                    r"위원회.*무엇", r"위원회.*어디", r"위원회.*설명",
                    r"조정.*신청.*기관", r"분쟁.*조정.*기관", r"신고.*기관",
                    r"업무.*담당", r"책임.*기관", r"주관.*기관"
                ],
                "context_keywords": [
                    "전자금융분쟁조정위원회", "개인정보보호위원회", "금융감독원",
                    "한국은행", "분쟁조정", "신고", "상담", "접수", "문의"
                ],
                "answer_indicators": ["에서", "가", "는", "담당", "수행", "관리"]
            },
            "특징_묻기": {
                "direct_patterns": [
                    r"특징.*설명", r"특징.*기술", r"특징.*무엇", r"특성.*설명",
                    r"성질.*설명", r"원리.*설명", r"방식.*설명", r"형태.*설명",
                    r"어떤.*특징", r"주요.*특징", r"핵심.*특징", r"중요.*특징"
                ],
                "context_keywords": [
                    "트로이", "악성코드", "원격제어", "딥페이크", "특징", "특성",
                    "성질", "원리", "방식", "형태", "구조", "기능"
                ],
                "answer_indicators": ["특징", "특성", "성질", "원리", "방식"]
            },
            "지표_묻기": {
                "direct_patterns": [
                    r"지표.*설명", r"지표.*기술", r"지표.*무엇", r"탐지.*지표",
                    r"징후.*설명", r"신호.*설명", r"패턴.*설명", r"모니터링.*지표",
                    r"어떤.*지표", r"주요.*지표", r"핵심.*지표", r"중요.*지표"
                ],
                "context_keywords": [
                    "탐지", "모니터링", "분석", "징후", "신호", "패턴", "지표",
                    "식별", "발견", "확인", "관찰", "추적"
                ],
                "answer_indicators": ["지표", "징후", "패턴", "탐지", "모니터링"]
            },
            "방안_묻기": {
                "direct_patterns": [
                    r"방안.*설명", r"방안.*기술", r"방안.*무엇", r"대책.*설명",
                    r"조치.*설명", r"대응.*방안", r"해결.*방안", r"관리.*방안",
                    r"어떤.*방안", r"주요.*방안", r"효과적.*방안", r"적절한.*방안"
                ],
                "context_keywords": [
                    "대응", "관리", "해결", "개선", "강화", "예방", "조치", "대책",
                    "방안", "절차", "체계", "시스템", "프로세스"
                ],
                "answer_indicators": ["방안", "대책", "조치", "관리", "대응"]
            },
            "절차_묻기": {
                "direct_patterns": [
                    r"절차.*설명", r"절차.*기술", r"절차.*무엇", r"과정.*설명",
                    r"단계.*설명", r"순서.*설명", r"프로세스.*설명",
                    r"어떤.*절차", r"주요.*절차", r"구체적.*절차"
                ],
                "context_keywords": [
                    "절차", "과정", "단계", "순서", "프로세스", "진행", "수행",
                    "실행", "처리", "진행방법", "수행방법"
                ],
                "answer_indicators": ["절차", "과정", "단계", "순서", "프로세스"]
            },
            "조치_묻기": {
                "direct_patterns": [
                    r"조치.*설명", r"조치.*기술", r"조치.*무엇", r"보안.*조치",
                    r"예방.*조치", r"대응.*조치", r"보완.*조치",
                    r"어떤.*조치", r"필요.*조치", r"적절한.*조치"
                ],
                "context_keywords": [
                    "조치", "보안", "예방", "대응", "보완", "강화", "개선",
                    "관리", "통제", "제어", "보호"
                ],
                "answer_indicators": ["조치", "보안", "예방", "대응", "관리"]
            }
        }

        # 컨텍스트 기반 가중치 설정
        self.context_weights = {
            "domain_match": 0.3,      # 도메인 일치
            "keyword_density": 0.3,   # 키워드 밀도  
            "pattern_strength": 0.4,  # 패턴 강도
        }

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

        # 기본 향상된 의도 패턴 설정
        self._setup_enhanced_intent_patterns()

    def analyze_question_intent(self, question: str) -> Dict:
        """향상된 질문 의도 분석"""
        question_lower = question.lower()

        intent_analysis = {
            "primary_intent": "일반",
            "intent_confidence": 0.0,
            "secondary_intents": [],
            "detected_patterns": [],
            "answer_type_required": "설명형",
            "context_hints": [],
            "quality_risk": False,
            "domain_context": "",
            "semantic_indicators": [],
        }

        # 1단계: 패턴 기반 의도 분석
        pattern_scores = self._analyze_intent_patterns(question_lower)
        
        # 2단계: 컨텍스트 기반 의도 분석  
        context_scores = self._analyze_intent_context(question, question_lower)
        
        # 3단계: 키워드 기반 의도 분석
        keyword_scores = self._analyze_intent_keywords(question_lower)
        
        # 4단계: 종합 점수 계산
        combined_scores = self._combine_intent_scores(
            pattern_scores, context_scores, keyword_scores
        )
        
        # 5단계: 최종 의도 결정
        if combined_scores:
            sorted_intents = sorted(
                combined_scores.items(), key=lambda x: x[1]["total_score"], reverse=True
            )
            
            best_intent = sorted_intents[0]
            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = best_intent[1]["total_score"]
            intent_analysis["detected_patterns"] = best_intent[1].get("patterns", [])
            
            # 부차적 의도들 기록
            if len(sorted_intents) > 1:
                intent_analysis["secondary_intents"] = [
                    {
                        "intent": intent,
                        "score": data["total_score"],
                        "confidence": data["total_score"]
                    }
                    for intent, data in sorted_intents[1:3]
                ]

            # 답변 유형 결정
            intent_analysis["answer_type_required"] = self._determine_answer_type(
                best_intent[0], question
            )
            
            # 컨텍스트 힌트 생성
            intent_analysis["context_hints"] = self._generate_context_hints(
                best_intent[0], question, best_intent[1]
            )
            
            # 도메인 컨텍스트 설정
            intent_analysis["domain_context"] = self._extract_domain_context(question)
            
            # 의미 지표 추출
            intent_analysis["semantic_indicators"] = self._extract_semantic_indicators(
                question, best_intent[0]
            )

        # 6단계: 품질 위험 평가
        intent_analysis["quality_risk"] = self._assess_quality_risk(question, intent_analysis)

        return intent_analysis

    def _analyze_intent_patterns(self, question_lower: str) -> Dict:
        """패턴 기반 의도 분석"""
        pattern_scores = {}
        
        for intent_type, intent_data in self.enhanced_intent_patterns.items():
            total_score = 0
            matched_patterns = []
            
            # 직접 패턴 매칭
            for pattern in intent_data["direct_patterns"]:
                matches = re.findall(pattern, question_lower)
                if matches:
                    score = len(matches) * 1.5  # 직접 패턴은 높은 가중치
                    total_score += score
                    matched_patterns.append(pattern)
            
            if total_score > 0:
                pattern_scores[intent_type] = {
                    "pattern_score": total_score,
                    "patterns": matched_patterns,
                    "match_count": len(matched_patterns)
                }
        
        return pattern_scores

    def _analyze_intent_context(self, question: str, question_lower: str) -> Dict:
        """컨텍스트 기반 의도 분석"""
        context_scores = {}
        
        for intent_type, intent_data in self.enhanced_intent_patterns.items():
            context_score = 0
            context_matches = []
            
            # 컨텍스트 키워드 분석
            for keyword in intent_data["context_keywords"]:
                if keyword.lower() in question_lower:
                    # 키워드 중요도에 따른 가중치
                    if len(keyword) > 5:  # 긴 키워드는 더 특화적
                        context_score += 2.0
                    else:
                        context_score += 1.0
                    context_matches.append(keyword)
            
            # 질문 길이에 따른 컨텍스트 보정
            if len(question) > 100 and context_score > 0:
                context_score *= 1.2  # 긴 질문에서는 컨텍스트를 더 중시
            
            if context_score > 0:
                context_scores[intent_type] = {
                    "context_score": context_score,
                    "context_matches": context_matches,
                    "keyword_density": len(context_matches) / len(intent_data["context_keywords"])
                }
        
        return context_scores

    def _analyze_intent_keywords(self, question_lower: str) -> Dict:
        """키워드 기반 의도 분석"""
        keyword_scores = {}
        
        # 추가적인 키워드 기반 분석
        additional_keywords = {
            "기관_묻기": {
                "primary": ["위원회", "기관", "담당", "어디", "어떤", "무엇"],
                "secondary": ["조정", "신고", "접수", "상담", "문의", "관리"],
                "weights": {"primary": 2.0, "secondary": 1.0}
            },
            "특징_묻기": {
                "primary": ["특징", "특성", "성질", "원리", "방식"],
                "secondary": ["구조", "기능", "형태", "성격", "속성"],
                "weights": {"primary": 2.0, "secondary": 1.0}
            },
            "지표_묻기": {
                "primary": ["지표", "징후", "신호", "패턴", "탐지"],
                "secondary": ["모니터링", "분석", "식별", "발견", "확인"],
                "weights": {"primary": 2.0, "secondary": 1.0}
            },
            "방안_묻기": {
                "primary": ["방안", "대책", "조치", "대응", "해결"],
                "secondary": ["관리", "개선", "강화", "예방", "보완"],
                "weights": {"primary": 2.0, "secondary": 1.0}
            },
            "절차_묻기": {
                "primary": ["절차", "과정", "단계", "순서", "프로세스"],
                "secondary": ["진행", "수행", "실행", "처리", "방법"],
                "weights": {"primary": 2.0, "secondary": 1.0}
            },
            "조치_묻기": {
                "primary": ["조치", "보안", "예방", "대응", "보완"],
                "secondary": ["강화", "개선", "관리", "통제", "보호"],
                "weights": {"primary": 2.0, "secondary": 1.0}
            }
        }
        
        for intent_type, keyword_data in additional_keywords.items():
            total_score = 0
            found_keywords = []
            
            # Primary 키워드 분석
            for keyword in keyword_data["primary"]:
                if keyword in question_lower:
                    total_score += keyword_data["weights"]["primary"]
                    found_keywords.append(f"primary:{keyword}")
            
            # Secondary 키워드 분석
            for keyword in keyword_data["secondary"]:
                if keyword in question_lower:
                    total_score += keyword_data["weights"]["secondary"]
                    found_keywords.append(f"secondary:{keyword}")
            
            if total_score > 0:
                keyword_scores[intent_type] = {
                    "keyword_score": total_score,
                    "found_keywords": found_keywords,
                    "keyword_coverage": len(found_keywords) / (len(keyword_data["primary"]) + len(keyword_data["secondary"]))
                }
        
        return keyword_scores

    def _combine_intent_scores(
        self, pattern_scores: Dict, context_scores: Dict, keyword_scores: Dict
    ) -> Dict:
        """의도 점수 종합"""
        combined_scores = {}
        
        # 모든 의도 유형 수집
        all_intent_types = set()
        all_intent_types.update(pattern_scores.keys())
        all_intent_types.update(context_scores.keys())
        all_intent_types.update(keyword_scores.keys())
        
        for intent_type in all_intent_types:
            # 각 점수 가져오기
            pattern_score = pattern_scores.get(intent_type, {}).get("pattern_score", 0)
            context_score = context_scores.get(intent_type, {}).get("context_score", 0)
            keyword_score = keyword_scores.get(intent_type, {}).get("keyword_score", 0)
            
            # 가중치 적용하여 총 점수 계산
            total_score = (
                pattern_score * self.context_weights["pattern_strength"] +
                context_score * self.context_weights["domain_match"] +
                keyword_score * self.context_weights["keyword_density"]
            )
            
            if total_score > 0:
                combined_scores[intent_type] = {
                    "total_score": min(total_score / 5.0, 1.0),  # 0-1 범위로 정규화
                    "pattern_score": pattern_score,
                    "context_score": context_score,
                    "keyword_score": keyword_score,
                    "patterns": pattern_scores.get(intent_type, {}).get("patterns", []),
                    "context_matches": context_scores.get(intent_type, {}).get("context_matches", []),
                    "found_keywords": keyword_scores.get(intent_type, {}).get("found_keywords", [])
                }
        
        return combined_scores

    def _determine_answer_type(self, intent_type: str, question: str) -> str:
        """답변 유형 결정"""
        question_lower = question.lower()
        
        if intent_type == "기관_묻기":
            return "기관명"
        elif intent_type == "특징_묻기":
            return "특징설명"
        elif intent_type == "지표_묻기":
            return "지표나열"
        elif intent_type == "방안_묻기":
            return "방안제시"
        elif intent_type == "절차_묻기":
            return "절차설명"
        elif intent_type == "조치_묻기":
            return "조치설명"
        elif any(word in question_lower for word in ["법령", "규정", "법률"]):
            return "법령설명"
        elif any(word in question_lower for word in ["정의", "개념", "의미"]):
            return "정의설명"
        else:
            return "설명형"

    def _generate_context_hints(self, intent_type: str, question: str, intent_data: Dict) -> List[str]:
        """컨텍스트 힌트 생성"""
        hints = []
        
        # 의도별 기본 힌트
        intent_hints = {
            "기관_묻기": ["구체적인 기관명 필요", "소속 기관 명시", "담당 업무 설명"],
            "특징_묻기": ["주요 특징 나열", "핵심 특성 설명", "다른 것과의 차이점"],
            "지표_묻기": ["탐지 지표 구체화", "모니터링 방법 설명", "식별 기준 제시"],
            "방안_묻기": ["구체적 실행방안", "단계별 대응책", "예방부터 복구까지"],
            "절차_묻기": ["단계별 순서", "각 단계의 요구사항", "프로세스 설명"],
            "조치_묻기": ["보안조치 내용", "기술적/관리적 조치", "예방 조치"]
        }
        
        hints.extend(intent_hints.get(intent_type, ["전문적 설명", "구체적 내용"]))
        
        # 컨텍스트 매칭 기반 추가 힌트
        context_matches = intent_data.get("context_matches", [])
        if "트로이" in context_matches or "악성코드" in context_matches:
            hints.append("악성코드 관련 기술적 설명 포함")
        if "전자금융" in context_matches:
            hints.append("전자금융거래법 관련 내용 포함")
        if "개인정보" in context_matches:
            hints.append("개인정보보호법 관련 내용 포함")
        
        return hints

    def _extract_domain_context(self, question: str) -> str:
        """도메인 컨텍스트 추출"""
        question_lower = question.lower()
        
        domain_indicators = {
            "사이버보안": ["트로이", "악성코드", "보안", "해킹", "침입"],
            "개인정보보호": ["개인정보", "정보주체", "개인정보보호법"],
            "전자금융": ["전자금융", "분쟁조정", "접근매체", "전자금융거래법"],
            "정보보안": ["정보보안", "관리체계", "isms"],
            "금융투자": ["금융투자", "자본시장법", "투자자문"],
            "위험관리": ["위험관리", "위험평가", "내부통제"]
        }
        
        for domain, indicators in domain_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                return domain
        
        return "일반"

    def _extract_semantic_indicators(self, question: str, intent_type: str) -> List[str]:
        """의미 지표 추출"""
        indicators = []
        question_lower = question.lower()
        
        # 의도별 의미 지표
        semantic_patterns = {
            "기관_묻기": ["담당", "관리", "소관", "업무", "역할", "기능"],
            "특징_묻기": ["특징", "특성", "성질", "원리", "구조", "기능"],
            "지표_묻기": ["탐지", "모니터링", "분석", "식별", "징후", "패턴"],
            "방안_묻기": ["대응", "해결", "관리", "개선", "강화", "예방"],
            "절차_묻기": ["단계", "순서", "과정", "절차", "프로세스", "진행"],
            "조치_묻기": ["조치", "대응", "예방", "보안", "보완", "강화"]
        }
        
        if intent_type in semantic_patterns:
            for pattern in semantic_patterns[intent_type]:
                if pattern in question_lower:
                    indicators.append(pattern)
        
        return indicators

    def _assess_quality_risk(self, question: str, intent_analysis: Dict) -> bool:
        """품질 위험 평가"""
        risk_factors = []
        
        # 낮은 의도 신뢰도
        if intent_analysis["intent_confidence"] < 0.3:
            risk_factors.append("low_confidence")
        
        # 복잡한 질문
        if len(question) > 300:
            risk_factors.append("complex_question")
        
        # 모호한 표현
        ambiguous_terms = ["관련", "해당", "적절한", "필요한"]
        if sum(1 for term in ambiguous_terms if term in question) > 2:
            risk_factors.append("ambiguous_terms")
        
        # 전문 용어 부족
        if not intent_analysis["semantic_indicators"]:
            risk_factors.append("lack_technical_terms")
        
        return len(risk_factors) >= 2

    def validate_answer_intent_match(
        self, answer: str, question: str, intent_analysis: Dict
    ) -> bool:
        """답변과 질문 의도 일치성 검증 - 강화된 버전"""
        if not answer or not intent_analysis:
            return False

        # 치명적인 반복 패턴이 있으면 즉시 실패
        if self.detect_critical_repetitive_patterns(answer):
            return False

        required_type = intent_analysis.get("answer_type_required", "설명형")
        answer_lower = answer.lower()
        primary_intent = intent_analysis.get("primary_intent", "일반")

        # 의도별 상세 검증
        validation_result = self._validate_by_intent_type(
            answer, answer_lower, required_type, primary_intent, question, intent_analysis
        )

        # 기본 검증이 실패해도 컨텍스트 기반 추가 검증
        if not validation_result:
            validation_result = self._context_based_validation(
                answer, answer_lower, question, intent_analysis
            )

        return validation_result

    def _validate_by_intent_type(
        self, answer: str, answer_lower: str, required_type: str, 
        primary_intent: str, question: str, intent_analysis: Dict
    ) -> bool:
        """의도 유형별 검증"""
        
        # 기관명이 필요한 경우 - 더 정교한 검증
        if required_type == "기관명" or "기관" in primary_intent:
            return self._validate_institution_answer(answer_lower, question, intent_analysis)
        
        # 특징 설명이 필요한 경우
        elif required_type == "특징설명" or "특징" in primary_intent:
            return self._validate_feature_answer(answer_lower, question, intent_analysis)
        
        # 지표 나열이 필요한 경우
        elif required_type == "지표나열" or "지표" in primary_intent:
            return self._validate_indicator_answer(answer_lower, question, intent_analysis)
        
        # 방안 제시가 필요한 경우
        elif required_type == "방안제시" or "방안" in primary_intent:
            return self._validate_solution_answer(answer_lower, question, intent_analysis)
        
        # 절차 설명이 필요한 경우
        elif required_type == "절차설명" or "절차" in primary_intent:
            return self._validate_procedure_answer(answer_lower, question, intent_analysis)
        
        # 조치 설명이 필요한 경우
        elif required_type == "조치설명" or "조치" in primary_intent:
            return self._validate_measure_answer(answer_lower, question, intent_analysis)
        
        # 기본적으로 통과 - 기준 완화
        else:
            return self._validate_general_answer(answer_lower, question, intent_analysis)

    def _validate_institution_answer(
        self, answer_lower: str, question: str, intent_analysis: Dict
    ) -> bool:
        """기관 답변 검증"""
        # 구체적인 기관명 확인
        specific_institutions = [
            "전자금융분쟁조정위원회", "금융감독원", "개인정보보호위원회",
            "개인정보침해신고센터", "한국은행", "금융위원회", "과학기술정보통신부"
        ]
        
        # 일반적인 기관 키워드
        general_institutions = [
            "위원회", "감독원", "은행", "기관", "센터", "청", "부", "원"
        ]
        
        # 구체적 기관명이 있으면 통과
        if any(inst in answer_lower for inst in specific_institutions):
            return True
        
        # 일반 기관 키워드가 있고 관련 업무 내용이 있으면 통과
        has_institution = any(keyword in answer_lower for keyword in general_institutions)
        related_work = any(word in answer_lower for word in [
            "담당", "업무", "수행", "관리", "조정", "신고", "접수", "상담"
        ])
        
        if has_institution and related_work:
            return True
        
        # 질문 컨텍스트와 매칭되는 내용이 있으면 통과
        if "전자금융" in question and any(word in answer_lower for word in ["전자금융", "분쟁", "조정"]):
            return True
        if "개인정보" in question and any(word in answer_lower for word in ["개인정보", "보호", "신고"]):
            return True
        if "한국은행" in question and any(word in answer_lower for word in ["한국은행", "자료제출", "통화"]):
            return True
        
        return False

    def _validate_feature_answer(
        self, answer_lower: str, question: str, intent_analysis: Dict
    ) -> bool:
        """특징 답변 검증"""
        feature_keywords = [
            "특징", "특성", "속성", "성질", "기능", "역할", "원리", "성격", "방식", "형태"
        ]
        
        descriptive_words = [
            "위장", "은밀", "지속", "제어", "접근", "수행", "활동", "동작", "실행",
            "생성", "변조", "탐지", "식별", "분석", "모니터링"
        ]
        
        # 특징 관련 키워드가 있거나
        has_feature_keywords = any(keyword in answer_lower for keyword in feature_keywords)
        
        # 설명적 내용이 충분하면 통과
        descriptive_count = sum(1 for word in descriptive_words if word in answer_lower)
        has_descriptive_content = descriptive_count >= 2
        
        # 질문 컨텍스트와 일치하는 내용
        context_match = False
        if "트로이" in question or "악성코드" in question:
            context_match = any(word in answer_lower for word in ["악성", "원격", "제어", "위장", "침투"])
        elif "딥페이크" in question:
            context_match = any(word in answer_lower for word in ["가짜", "인공지능", "생성", "조작"])
        
        return has_feature_keywords or has_descriptive_content or context_match

    def _validate_indicator_answer(
        self, answer_lower: str, question: str, intent_analysis: Dict
    ) -> bool:
        """지표 답변 검증"""
        indicator_keywords = [
            "지표", "신호", "징후", "패턴", "행동", "활동", "모니터링", "탐지",
            "발견", "식별", "분석", "확인", "추적", "관찰"
        ]
        
        technical_indicators = [
            "네트워크", "트래픽", "프로세스", "파일", "시스템", "로그", "연결",
            "접근", "변경", "사용", "성능", "리소스", "메모리", "cpu"
        ]
        
        # 지표 관련 키워드가 있거나
        has_indicator_keywords = any(keyword in answer_lower for keyword in indicator_keywords)
        
        # 기술적 지표들이 언급되면 통과
        technical_count = sum(1 for word in technical_indicators if word in answer_lower)
        has_technical_indicators = technical_count >= 2
        
        # 탐지/모니터링 관련 내용
        monitoring_content = any(word in answer_lower for word in [
            "실시간", "모니터링", "분석", "추적", "감시", "점검"
        ])
        
        return has_indicator_keywords or has_technical_indicators or monitoring_content

    def _validate_solution_answer(
        self, answer_lower: str, question: str, intent_analysis: Dict
    ) -> bool:
        """방안 답변 검증"""
        solution_keywords = [
            "방안", "대책", "조치", "해결", "대응", "관리", "처리", "절차",
            "개선", "예방", "보완", "강화", "구축", "수립", "운영"
        ]
        
        action_words = [
            "수립", "구축", "시행", "실시", "강화", "개선", "마련", "도입",
            "운영", "적용", "활용", "실행", "수행", "추진"
        ]
        
        # 방안 관련 키워드가 있거나
        has_solution_keywords = any(keyword in answer_lower for keyword in solution_keywords)
        
        # 실행 가능한 내용이 있으면 통과
        action_count = sum(1 for word in action_words if word in answer_lower)
        has_actionable_content = action_count >= 2
        
        # "해야" 구문이 있고 충분한 길이면 통과
        has_requirement = "해야" in answer_lower and len(answer_lower) >= 30
        
        return has_solution_keywords or has_actionable_content or has_requirement

    def _validate_procedure_answer(
        self, answer_lower: str, question: str, intent_analysis: Dict
    ) -> bool:
        """절차 답변 검증"""
        procedure_keywords = [
            "절차", "과정", "단계", "순서", "프로세스", "진행", "수행",
            "실행", "처리", "방법", "방식"
        ]
        
        step_indicators = [
            "첫째", "둘째", "셋째", "먼저", "다음", "마지막", "단계적",
            "순차적", "차례대로", "단계", "과정"
        ]
        
        # 절차 관련 키워드가 있거나
        has_procedure_keywords = any(keyword in answer_lower for keyword in procedure_keywords)
        
        # 단계 표시가 있으면 통과
        has_step_indicators = any(indicator in answer_lower for indicator in step_indicators)
        
        # 순서나 과정을 나타내는 구두점
        has_sequence_markers = "," in answer_lower or "→" in answer_lower or "단계" in answer_lower
        
        return has_procedure_keywords or has_step_indicators or has_sequence_markers

    def _validate_measure_answer(
        self, answer_lower: str, question: str, intent_analysis: Dict
    ) -> bool:
        """조치 답변 검증"""
        measure_keywords = [
            "조치", "대응", "대책", "방안", "보안", "예방", "개선", "강화",
            "보완", "관리", "통제", "제어", "보호"
        ]
        
        # 조치 관련 키워드가 있거나
        has_measure_keywords = any(keyword in answer_lower for keyword in measure_keywords)
        
        # "필요" 구문이 있고 관련 내용이 있으면 통과
        has_necessity = "필요" in answer_lower and any(word in answer_lower for word in [
            "보안", "관리", "예방", "대응", "조치"
        ])
        
        return has_measure_keywords or has_necessity

    def _validate_general_answer(
        self, answer_lower: str, question: str, intent_analysis: Dict
    ) -> bool:
        """일반 답변 검증"""
        # 최소한의 의미있는 키워드가 있으면 통과
        meaningful_keywords = [
            "법령", "규정", "관리", "조치", "절차", "기준", "정책", "체계",
            "시스템", "필요", "중요", "수행", "실시", "구축", "운영"
        ]
        
        keyword_count = sum(1 for word in meaningful_keywords if word in answer_lower)
        
        # 길이와 키워드 수를 종합 평가
        length_ok = len(answer_lower) >= 25
        content_ok = keyword_count >= 2
        
        return length_ok and content_ok

    def _context_based_validation(
        self, answer: str, answer_lower: str, question: str, intent_analysis: Dict
    ) -> bool:
        """컨텍스트 기반 추가 검증"""
        
        # 질문-답변 컨텍스트 매칭
        question_lower = question.lower()
        
        context_matches = []
        
        # 도메인 매칭
        domain_context = intent_analysis.get("domain_context", "")
        if domain_context and domain_context != "일반":
            domain_keywords = self.domain_keywords.get(domain_context, [])
            domain_match_count = sum(1 for keyword in domain_keywords if keyword in answer_lower)
            if domain_match_count > 0:
                context_matches.append("domain_match")
        
        # 의미 지표 매칭
        semantic_indicators = intent_analysis.get("semantic_indicators", [])
        semantic_match_count = sum(1 for indicator in semantic_indicators if indicator in answer_lower)
        if semantic_match_count > 0:
            context_matches.append("semantic_match")
        
        # 질문 키워드 매칭
        question_keywords = re.findall(r'[가-힣]{2,}', question_lower)
        answer_keywords = re.findall(r'[가-힣]{2,}', answer_lower)
        
        common_keywords = set(question_keywords) & set(answer_keywords)
        if len(common_keywords) >= 2:
            context_matches.append("keyword_overlap")
        
        # 컨텍스트 매칭이 충분하면 통과
        return len(context_matches) >= 2

    def detect_critical_repetitive_patterns(self, text: str) -> bool:
        """치명적인 반복 패턴 감지 - 스마트 감지"""
        if not text or len(text) < 20:
            return False

        # 1단계: 명백한 문제 패턴
        critical_patterns = [
            r"갈취 묻는 말",
            r"묻고 갈취",
            r"(.{1,3})\s*(\1\s*){10,}",  # 10회 이상 반복
        ]

        for pattern in critical_patterns:
            if re.search(pattern, text):
                return True

        # 2단계: 의미 단위 반복 감지
        sentences = text.split('.')
        if len(sentences) > 2:
            # 같은 문장이 3번 이상 반복되는 경우
            for sentence in sentences:
                if sentence.strip() and len(sentence) > 15:
                    count = sentences.count(sentence)
                    if count >= 3:
                        return True

        # 3단계: 단어 수준 심각한 반복
        words = text.split()
        if len(words) > 10:
            for word in set(words):
                if len(word) > 2 and words.count(word) >= 8:  # 8회 이상만
                    return True

        return False

    # 기존 메서드들 유지 (호환성을 위해)
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
                        "개인정보보호법", "전자금융거래법", "자본시장법", "ISMS",
                        "트로이", "RAT", "원격제어", "분쟁조정", "위험관리",
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
        return detected_domain

    def clean_korean_text(self, text: str) -> str:
        """한국어 전용 텍스트 정리"""
        if not text:
            return ""

        # 치명적인 반복 패턴만 조기 감지 및 제거
        if self.detect_critical_repetitive_patterns(text):
            text = self._remove_critical_repetitive_patterns(text)
            if len(text) < 10:
                return "텍스트 정리 중 내용이 부족합니다."

        # 깨진 문자 복구
        text = self._restore_korean_characters(text)

        # 텍스트 품질 향상
        text = self._enhance_korean_text_quality(text)

        # 문법 구조 개선
        text = self.fix_grammatical_structure(text)

        return text

    def _remove_critical_repetitive_patterns(self, text: str) -> str:
        """치명적인 반복 패턴 제거"""
        if not text:
            return ""

        # 문제가 되는 특정 패턴만 제거
        problematic_removals = [
            "갈취 묻는 말",
            "묻고 갈취",
        ]

        for pattern in problematic_removals:
            text = text.replace(pattern, "")

        # 의미 단위별 중복 제거
        sentences = text.split('.')
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if sentence_clean and len(sentence_clean) > 5:
                # 중복 문장 제거
                if sentence_clean not in seen_sentences:
                    unique_sentences.append(sentence_clean)
                    seen_sentences.add(sentence_clean)

        # 재조립
        text = '. '.join(unique_sentences)
        if text and not text.endswith('.'):
            text += '.'

        # 단어 수준 반복 제거 - 관대하게
        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            current_word = words[i]
            count = 1
            
            # 연속된 동일 단어 계산
            while i + count < len(words) and words[i + count] == current_word:
                count += 1
            
            # 단어 길이에 따른 허용 개수 조정
            if len(current_word) <= 2:
                # 조사, 어미 등은 3개까지 허용
                cleaned_words.extend([current_word] * min(3, count))
            elif len(current_word) <= 5:
                # 일반 단어는 2개까지 허용
                cleaned_words.extend([current_word] * min(2, count))
            else:
                # 긴 단어는 1개만 허용
                cleaned_words.append(current_word)
            
            i += count

        text = " ".join(cleaned_words)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _restore_korean_characters(self, text: str) -> str:
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

    def _enhance_korean_text_quality(self, text: str) -> str:
        """한국어 텍스트 품질 향상"""
        if not text:
            return ""

        # 품질 패턴 적용
        for pattern_config in self.korean_quality_patterns:
            pattern = pattern_config["pattern"]
            replacement = pattern_config["replacement"]
            text = re.sub(pattern, replacement, text)

        # 의미없는 문자 제거 - 더 관대하게
        text = re.sub(r"[^\w\s가-힣.,!?()[\]\-:;/·]", " ", text)

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
        ]

        for pattern, replacement in grammar_fixes:
            text = re.sub(pattern, replacement, text)

        # 마지막 마침표 처리
        if text and not text.endswith("."):
            text += "."

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

    def validate_korean_answer(
        self, answer: str, question_type: str, max_choice: int = 5, question: str = ""
    ) -> bool:
        """한국어 답변 유효성 검증 - 강화된 버전"""
        if not answer:
            return False

        answer = str(answer).strip()

        # 치명적인 반복 패턴만 조기 감지
        if self.detect_critical_repetitive_patterns(answer):
            return False

        if question_type == "multiple_choice":
            # 객관식: 지정된 범위의 숫자
            if not answer.isdigit() or not (1 <= int(answer) <= max_choice):
                return False
            return True

        else:
            # 주관식: 한국어 전용 검증
            clean_answer = self.clean_korean_text(answer)

            # 정리 후 치명적인 반복 패턴만 재확인
            if self.detect_critical_repetitive_patterns(clean_answer):
                return False

            # 길이 검증
            if not (
                self.korean_requirements["min_length"]
                <= len(clean_answer)
                <= self.korean_requirements["max_length"]
            ):
                return False

            # 한국어 비율 검증
            korean_ratio = self.calculate_korean_ratio(clean_answer)
            if korean_ratio < self.korean_requirements["min_korean_ratio"]:
                return False

            # 의미 있는 내용인지 확인
            meaningful_keywords = [
                "법령", "규정", "조치", "관리", "보안", "방안", "절차", "기준",
                "정책", "체계", "시스템", "통제", "특징", "지표", "탐지", "대응",
                "기관", "위원회", "필요", "중요", "수행", "실시", "구축", "운영"
            ]
            if not any(word in clean_answer for word in meaningful_keywords):
                return False

            return True

    def analyze_question_difficulty(self, question: str) -> str:
        """질문 난이도 분석"""
        question_lower = question.lower()

        # 전문 용어 개수
        technical_terms = [
            "isms", "pims", "sbom", "원격제어", "침입탐지", "트로이", "멀웨어",
            "랜섬웨어", "딥페이크", "피싱", "접근매체", "전자서명", "개인정보보호법",
            "자본시장법", "rat", "원격접근", "탐지지표", "apt", "ddos", "ids", "ips",
            "bcp", "drp", "isms-p", "분쟁조정", "금융투자업", "위험관리", "재해복구"
        ]

        term_count = sum(1 for term in technical_terms if term in question_lower)

        # 문장 길이
        length = len(question)

        # 선택지 개수 (객관식인 경우)
        choice_count = len(self._extract_choices(question))

        # 난이도 계산
        if term_count >= 3 or length > 400 or choice_count >= 5:
            return "고급"
        elif term_count >= 1 or length > 200 or choice_count >= 4:
            return "중급"
        else:
            return "초급"

    def _extract_choices(self, question: str) -> List[str]:
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

        return choices[:5]

    def normalize_korean_answer(
        self, answer: str, question_type: str, max_choice: int = 5
    ) -> str:
        """한국어 답변 정규화"""
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
                return "답변 생성 중 반복 패턴이 감지되어 재생성이 필요합니다."

            # 의미 없는 짧은 문장 제거
            if len(answer) < self.korean_requirements["min_length"]:
                return "답변 길이가 부족하여 생성에 실패했습니다."

            # 길이 제한
            if len(answer) > self.korean_requirements["max_length"]:
                sentences = answer.split(". ")
                answer = ". ".join(sentences[:4])
                if not answer.endswith("."):
                    answer += "."

            # 마침표 확인
            if answer and not answer.endswith((".", "다", "요", "함")):
                answer += "."

            return answer

    def cleanup(self):
        """정리"""
        pass