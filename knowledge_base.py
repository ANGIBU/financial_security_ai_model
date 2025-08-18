# knowledge_base.py

"""
금융보안 지식베이스
- 도메인별 키워드 분류
- 전문 용어 처리
- 한국어 전용 답변 템플릿 힌트 제공
- 대회 규칙 준수 검증
- 질문 의도별 지식 제공
"""

import pickle
import os
import re
import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import random

# 설정 파일 import
from config import JSON_CONFIG_FILES, TEMPLATE_QUALITY_CRITERIA


class FinancialSecurityKnowledgeBase:
    """금융보안 지식베이스"""

    def __init__(self):
        # pkl 저장 폴더 생성
        self.pkl_dir = Path("./pkl")
        self.pkl_dir.mkdir(exist_ok=True)

        # JSON 설정 파일 로드
        self._load_json_configs()

        # 템플릿 품질 평가 기준 (config.py에서 로드)
        self.template_quality_criteria = TEMPLATE_QUALITY_CRITERIA

        # 질문 분석 이력
        self.analysis_history = {
            "domain_frequency": {},
            "complexity_distribution": {},
            "question_patterns": [],
            "compliance_check": {
                "korean_only": 0,
                "law_references": 0,
                "technical_terms": 0,
            },
            "intent_analysis_history": {},
            "template_usage_stats": {},
            "template_effectiveness": {},
            "mc_pattern_accuracy": {},
            "institution_question_accuracy": {},
            "hint_provision_stats": {},
        }

        # 이전 분석 이력 로드
        self._load_analysis_history()

    def _load_json_configs(self):
        """JSON 설정 파일들 로드"""
        try:
            # knowledge_data.json 로드
            with open(JSON_CONFIG_FILES["knowledge_data"], "r", encoding="utf-8") as f:
                knowledge_data = json.load(f)

            # 지식베이스 데이터 할당
            self.korean_subjective_templates = knowledge_data[
                "korean_subjective_templates"
            ]
            self.domain_keywords = knowledge_data["domain_keywords"]
            self.korean_financial_terms = knowledge_data["korean_financial_terms"]
            self.institution_database = knowledge_data["institution_database"]
            self.mc_answer_patterns = knowledge_data["mc_answer_patterns"]

            print("지식베이스 설정 파일 로드 완료")

        except FileNotFoundError as e:
            print(f"설정 파일을 찾을 수 없습니다: {e}")
            self._load_default_configs()
        except json.JSONDecodeError as e:
            print(f"JSON 파일 파싱 오류: {e}")
            self._load_default_configs()
        except Exception as e:
            print(f"설정 파일 로드 중 오류: {e}")
            self._load_default_configs()

    def _load_default_configs(self):
        """기본 설정 로드 (JSON 파일 로드 실패 시)"""
        print("기본 설정으로 대체합니다.")

        # 최소한의 기본 설정
        self.korean_subjective_templates = {
            "일반": {
                "일반": [
                    "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
                ]
            }
        }

        self.domain_keywords = {"일반": ["법령", "규정", "관리", "조치", "절차"]}

        self.korean_financial_terms = {}
        self.institution_database = {}
        self.mc_answer_patterns = {}

    def _load_analysis_history(self):
        """이전 분석 이력 로드"""
        history_file = self.pkl_dir / "analysis_history.pkl"

        if history_file.exists():
            try:
                with open(history_file, "rb") as f:
                    saved_history = pickle.load(f)
                    self.analysis_history.update(saved_history)
            except Exception:
                pass

    def _save_analysis_history(self):
        """분석 이력 저장"""
        history_file = self.pkl_dir / "analysis_history.pkl"

        try:
            save_data = {
                **self.analysis_history,
                "last_updated": datetime.now().isoformat(),
            }

            # 최근 1000개 패턴만 저장
            save_data["question_patterns"] = save_data["question_patterns"][-1000:]

            with open(history_file, "wb") as f:
                pickle.dump(save_data, f)
        except Exception:
            pass

    def analyze_question(self, question: str) -> Dict:
        """질문 분석"""
        question_lower = question.lower()

        # 도메인 찾기
        detected_domains = []
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 핵심 키워드 가중치 적용
                    if keyword in [
                        "트로이",
                        "RAT",
                        "원격제어",
                        "SBOM",
                        "전자금융분쟁조정위원회",
                        "개인정보보호위원회",
                        "만 14세",
                        "위험 관리",
                        "금융투자업",
                    ]:
                        score += 3
                    else:
                        score += 1

            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            # 가장 높은 점수의 도메인 선택
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            detected_domains = [best_domain]
        else:
            detected_domains = ["일반"]

        # 복잡도 계산
        complexity = self._calculate_complexity(question)

        # 한국어 전문 용어 포함 여부
        korean_terms = self._find_korean_technical_terms(question)

        # 대회 규칙 준수 확인
        compliance_check = self._check_competition_compliance(question)

        # 기관 관련 질문인지 확인
        institution_info = self._check_institution_question(question)

        # 객관식 패턴 매칭
        mc_pattern_info = self._analyze_mc_pattern(question)

        # 분석 결과 저장
        analysis_result = {
            "domain": detected_domains,
            "complexity": complexity,
            "technical_level": self._determine_technical_level(
                complexity, korean_terms
            ),
            "korean_technical_terms": korean_terms,
            "compliance": compliance_check,
            "institution_info": institution_info,
            "mc_pattern_info": mc_pattern_info,
        }

        # 이력에 추가
        self._add_to_analysis_history(question, analysis_result)

        return analysis_result

    def _analyze_mc_pattern(self, question: str) -> Dict:
        """객관식 패턴 분석"""
        question_lower = question.lower()

        pattern_info = {
            "is_mc_question": False,
            "pattern_type": None,
            "pattern_confidence": 0.0,
            "pattern_key": None,
            "hint_available": False,
        }

        # 실제 데이터 패턴 매칭
        for pattern_key, pattern_data in self.mc_answer_patterns.items():
            keyword_matches = sum(
                1
                for keyword in pattern_data["question_keywords"]
                if keyword in question_lower
            )

            if keyword_matches >= 2:
                pattern_info["is_mc_question"] = True
                pattern_info["pattern_type"] = pattern_key
                pattern_info["pattern_confidence"] = keyword_matches / len(
                    pattern_data["question_keywords"]
                )
                pattern_info["pattern_key"] = pattern_key
                pattern_info["hint_available"] = True
                break

        return pattern_info

    def _check_institution_question(self, question: str) -> Dict:
        """기관 관련 질문 확인"""
        question_lower = question.lower()

        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None,
            "confidence": 0.0,
            "question_pattern": None,
            "hint_available": False,
        }

        # 기관 질문 패턴 확인
        institution_patterns = [
            "기관.*기술하세요",
            "기관.*설명하세요",
            "어떤.*기관",
            "어느.*기관",
            "조정.*신청.*기관",
            "분쟁.*조정.*기관",
            "신청.*수.*있는.*기관",
            "담당.*기관",
            "관리.*기관",
            "감독.*기관",
            "소관.*기관",
            "신고.*기관",
            "접수.*기관",
            "상담.*기관",
            "문의.*기관",
            "위원회.*무엇",
            "위원회.*어디",
            "위원회.*설명",
        ]

        pattern_matches = 0
        matched_pattern = None
        for pattern in institution_patterns:
            if re.search(pattern, question_lower):
                pattern_matches += 1
                matched_pattern = pattern

        is_asking_institution = pattern_matches > 0

        if is_asking_institution:
            institution_info["is_institution_question"] = True
            institution_info["confidence"] = min(pattern_matches / 2, 1.0)
            institution_info["question_pattern"] = matched_pattern
            institution_info["hint_available"] = True

            # 분야별 기관 확인
            for institution_key, institution_data in self.institution_database.items():
                if "관련질문패턴" in institution_data:
                    pattern_score = sum(
                        1
                        for pattern in institution_data["관련질문패턴"]
                        if pattern.lower() in question_lower
                    )

                    if pattern_score > 0:
                        institution_info["institution_type"] = institution_key
                        institution_info["relevant_institution"] = institution_data
                        institution_info["confidence"] = min(
                            pattern_score / len(institution_data["관련질문패턴"]), 1.0
                        )
                        break

            # 기존 로직으로 폴백
            if not institution_info["institution_type"]:
                if (
                    any(word in question_lower for word in ["전자금융", "전자적"])
                    and "분쟁" in question_lower
                ):
                    institution_info["institution_type"] = "전자금융분쟁조정"
                    institution_info["relevant_institution"] = (
                        self.institution_database.get("전자금융분쟁조정", {})
                    )
                elif any(word in question_lower for word in ["개인정보", "정보주체"]):
                    institution_info["institution_type"] = "개인정보보호"
                    institution_info["relevant_institution"] = (
                        self.institution_database.get("개인정보보호", {})
                    )
                elif (
                    any(
                        word in question_lower
                        for word in ["금융투자", "투자자문", "자본시장"]
                    )
                    and "분쟁" in question_lower
                ):
                    institution_info["institution_type"] = "금융투자분쟁조정"
                    institution_info["relevant_institution"] = (
                        self.institution_database.get("금융투자분쟁조정", {})
                    )
                elif any(
                    word in question_lower
                    for word in ["한국은행", "금융통화위원회", "자료제출"]
                ):
                    institution_info["institution_type"] = "한국은행"
                    institution_info["relevant_institution"] = (
                        self.institution_database.get("한국은행", {})
                    )

        return institution_info

    def _check_competition_compliance(self, question: str) -> Dict:
        """대회 규칙 준수 확인"""
        compliance = {
            "korean_content": True,
            "appropriate_domain": True,
            "no_external_dependency": True,
        }

        # 한국어 비율 확인
        korean_chars = len(
            [c for c in question if ord(c) >= 0xAC00 and ord(c) <= 0xD7A3]
        )
        total_chars = len([c for c in question if c.isalpha()])

        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            compliance["korean_content"] = korean_ratio > 0.7

        # 도메인 적절성 확인
        found_domains = []
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in question.lower() for keyword in keywords):
                found_domains.append(domain)

        compliance["appropriate_domain"] = len(found_domains) > 0

        return compliance

    def _add_to_analysis_history(self, question: str, analysis: Dict):
        """분석 이력에 추가"""
        # 도메인 빈도 업데이트
        for domain in analysis["domain"]:
            self.analysis_history["domain_frequency"][domain] = (
                self.analysis_history["domain_frequency"].get(domain, 0) + 1
            )

        # 복잡도 분포 업데이트
        level = analysis["technical_level"]
        self.analysis_history["complexity_distribution"][level] = (
            self.analysis_history["complexity_distribution"].get(level, 0) + 1
        )

        # 준수성 확인 업데이트
        if analysis["compliance"]["korean_content"]:
            self.analysis_history["compliance_check"]["korean_only"] += 1

        if any("법" in term for term in analysis["korean_technical_terms"]):
            self.analysis_history["compliance_check"]["law_references"] += 1

        if len(analysis["korean_technical_terms"]) > 0:
            self.analysis_history["compliance_check"]["technical_terms"] += 1

        # 기관 질문 이력 추가
        if analysis["institution_info"]["is_institution_question"]:
            institution_type = analysis["institution_info"]["institution_type"]
            if institution_type:
                if (
                    institution_type
                    not in self.analysis_history["institution_question_accuracy"]
                ):
                    self.analysis_history["institution_question_accuracy"][
                        institution_type
                    ] = {"total": 0, "high_confidence": 0}

                self.analysis_history["institution_question_accuracy"][
                    institution_type
                ]["total"] += 1
                if analysis["institution_info"]["confidence"] > 0.7:
                    self.analysis_history["institution_question_accuracy"][
                        institution_type
                    ]["high_confidence"] += 1

        # 객관식 패턴 정확도 추가
        if analysis["mc_pattern_info"]["is_mc_question"]:
            pattern_key = analysis["mc_pattern_info"]["pattern_key"]
            if pattern_key:
                if pattern_key not in self.analysis_history["mc_pattern_accuracy"]:
                    self.analysis_history["mc_pattern_accuracy"][pattern_key] = {
                        "total": 0,
                        "high_confidence": 0,
                    }

                self.analysis_history["mc_pattern_accuracy"][pattern_key]["total"] += 1
                if analysis["mc_pattern_info"]["pattern_confidence"] > 0.7:
                    self.analysis_history["mc_pattern_accuracy"][pattern_key][
                        "high_confidence"
                    ] += 1

        # 질문 패턴 추가
        pattern = {
            "question_length": len(question),
            "domain": analysis["domain"][0] if analysis["domain"] else "일반",
            "complexity": analysis["complexity"],
            "korean_terms_count": len(analysis["korean_technical_terms"]),
            "compliance_score": sum(analysis["compliance"].values())
            / len(analysis["compliance"]),
            "is_institution_question": analysis["institution_info"][
                "is_institution_question"
            ],
            "is_mc_pattern": analysis["mc_pattern_info"]["is_mc_question"],
            "timestamp": datetime.now().isoformat(),
        }

        self.analysis_history["question_patterns"].append(pattern)

    def get_mc_pattern_hints(self, question: str) -> str:
        """객관식 패턴 힌트 반환 - 직접 답변 대신 힌트 정보 제공"""
        mc_pattern_info = self._analyze_mc_pattern(question)

        if (
            mc_pattern_info["is_mc_question"]
            and mc_pattern_info["pattern_confidence"] > 0.5
        ):
            pattern_key = mc_pattern_info["pattern_key"]
            if pattern_key in self.mc_answer_patterns:
                pattern_data = self.mc_answer_patterns[pattern_key]

                # 힌트 통계 업데이트
                if (
                    "mc_pattern_hints"
                    not in self.analysis_history["hint_provision_stats"]
                ):
                    self.analysis_history["hint_provision_stats"][
                        "mc_pattern_hints"
                    ] = 0
                self.analysis_history["hint_provision_stats"]["mc_pattern_hints"] += 1

                # 설명 정보를 힌트로 제공
                hint_info = f"이 문제는 {pattern_data.get('explanation', '관련 내용')}에 대한 문제입니다."
                if "choices" in pattern_data:
                    hint_info += (
                        f" 선택지는 {', '.join(pattern_data['choices'])}입니다."
                    )

                return hint_info

        return None

    def get_template_hints(self, domain: str, intent_type: str = "일반") -> str:
        """템플릿 힌트 반환 - 직접 답변 대신 힌트 정보 제공"""

        # 템플릿 힌트 사용 통계 업데이트
        template_key = f"{domain}_{intent_type}"
        if template_key not in self.analysis_history["template_usage_stats"]:
            self.analysis_history["template_usage_stats"][template_key] = 0
        self.analysis_history["template_usage_stats"][template_key] += 1

        # 힌트 통계 업데이트
        if "template_hints" not in self.analysis_history["hint_provision_stats"]:
            self.analysis_history["hint_provision_stats"]["template_hints"] = 0
        self.analysis_history["hint_provision_stats"]["template_hints"] += 1

        # 도메인과 의도에 맞는 힌트 정보 생성
        if domain in self.korean_subjective_templates:
            domain_templates = self.korean_subjective_templates[domain]

            # 의도별 템플릿이 있는지 확인
            if isinstance(domain_templates, dict):
                if intent_type in domain_templates:
                    templates = domain_templates[intent_type]
                elif "일반" in domain_templates:
                    templates = domain_templates["일반"]
                else:
                    # dict의 첫 번째 값 사용
                    templates = list(domain_templates.values())[0]
            else:
                templates = domain_templates
        else:
            # 일반 템플릿 사용
            if "일반" in self.korean_subjective_templates:
                templates = self.korean_subjective_templates["일반"]["일반"]
            else:
                return None

        # 템플릿에서 핵심 키워드와 구조 추출하여 힌트 생성
        if isinstance(templates, list) and len(templates) > 0:
            sample_template = templates[0]

            # 핵심 키워드 추출
            key_phrases = []
            if "법령" in sample_template:
                key_phrases.append("관련 법령")
            if "규정" in sample_template:
                key_phrases.append("규정")
            if "관리" in sample_template:
                key_phrases.append("관리 방안")
            if "조치" in sample_template:
                key_phrases.append("보안조치")
            if "절차" in sample_template:
                key_phrases.append("절차")
            if "기관" in sample_template:
                key_phrases.append("관련 기관")

            # 힌트 정보 생성
            if key_phrases:
                hint_info = f"{intent_type}에 대한 답변 시 다음 요소들을 포함하세요: {', '.join(key_phrases)}"
            else:
                hint_info = f"{domain} 분야의 {intent_type} 관련 전문적인 내용을 포함하여 답변하세요."

            # 답변 구조 힌트 추가
            if intent_type == "기관_묻기":
                hint_info += " 구체적인 기관명과 소속을 명시하세요."
            elif intent_type == "특징_묻기":
                hint_info += " 주요 특징을 체계적으로 나열하세요."
            elif intent_type == "지표_묻기":
                hint_info += " 탐지 지표와 모니터링 방법을 설명하세요."
            elif intent_type == "방안_묻기":
                hint_info += " 실무적이고 구체적인 대응방안을 제시하세요."
            elif intent_type == "절차_묻기":
                hint_info += " 단계별 절차를 순서대로 설명하세요."
            elif intent_type == "조치_묻기":
                hint_info += " 구체적인 보안조치와 대응조치를 설명하세요."

            return hint_info

        return None

    def get_institution_hints(self, institution_type: str) -> str:
        """기관별 힌트 정보 반환 - 직접 답변 대신 힌트 정보 제공"""
        if institution_type in self.institution_database:
            info = self.institution_database[institution_type]

            # 힌트 통계 업데이트
            if "institution_hints" not in self.analysis_history["hint_provision_stats"]:
                self.analysis_history["hint_provision_stats"]["institution_hints"] = 0
            self.analysis_history["hint_provision_stats"]["institution_hints"] += 1

            # 기관 정보를 힌트로 제공
            hint_parts = []

            if "기관명" in info:
                hint_parts.append(f"기관명: {info['기관명']}")

            if "소속" in info:
                hint_parts.append(f"소속: {info['소속']}")

            if "역할" in info:
                hint_parts.append(f"주요 역할: {info['역할']}")

            if "근거법" in info:
                hint_parts.append(f"근거 법령: {info['근거법']}")

            if institution_type == "전자금융분쟁조정":
                hint_parts.append("전자금융거래 관련 분쟁조정 업무를 담당합니다.")
            elif institution_type == "개인정보보호":
                hint_parts.append("개인정보 보호 정책 수립과 감시 업무를 수행합니다.")
            elif institution_type == "금융투자분쟁조정":
                hint_parts.append("금융투자 관련 분쟁조정 업무를 담당합니다.")
            elif institution_type == "한국은행":
                hint_parts.append("통화신용정책 수행과 지급결제제도 운영을 담당합니다.")

            return " ".join(hint_parts)

        # 기본 힌트
        return "해당 분야의 전문 기관에서 관련 업무를 담당하고 있습니다."

    def get_korean_subjective_template(
        self, domain: str, intent_type: str = "일반"
    ) -> str:
        """한국어 주관식 답변 템플릿 반환 - 힌트용으로 변경"""
        # 이 메서드는 이제 힌트 정보만 반환
        return self.get_template_hints(domain, intent_type)

    def get_high_quality_template(
        self, domain: str, intent_type: str, min_quality: float = 0.8
    ) -> str:
        """고품질 템플릿 반환 - 힌트용으로 변경"""
        template_key = f"{domain}_{intent_type}"

        # 효과성이 검증된 템플릿 힌트 우선 사용
        if template_key in self.analysis_history["template_effectiveness"]:
            effectiveness = self.analysis_history["template_effectiveness"][
                template_key
            ]
            if (
                effectiveness["korean_ratio"] >= min_quality
                and effectiveness["usage_count"] >= 5
            ):
                # 검증된 고품질 템플릿의 힌트 정보 제공
                return self.get_template_hints(domain, intent_type)

        # 기본 템플릿 힌트 반환
        return self.get_template_hints(domain, intent_type)

    def get_subjective_template(self, domain: str, intent_type: str = "일반") -> str:
        """주관식 답변 템플릿 반환 - 힌트용으로 변경"""
        return self.get_template_hints(domain, intent_type)

    def _calculate_complexity(self, question: str) -> float:
        """질문 복잡도 계산"""
        # 길이 기반 복잡도
        length_factor = min(len(question) / 200, 1.0)

        # 한국어 전문 용어 개수
        korean_term_count = sum(
            1 for term in self.korean_financial_terms.keys() if term in question
        )
        term_factor = min(korean_term_count / 3, 1.0)

        # 도메인 개수
        domain_count = sum(
            1
            for keywords in self.domain_keywords.values()
            if any(keyword in question.lower() for keyword in keywords)
        )
        domain_factor = min(domain_count / 2, 1.0)

        return (length_factor + term_factor + domain_factor) / 3

    def _find_korean_technical_terms(self, question: str) -> List[str]:
        """한국어 전문 용어 찾기"""
        found_terms = []

        for term in self.korean_financial_terms.keys():
            if term in question:
                found_terms.append(term)

        return found_terms

    def _determine_technical_level(
        self, complexity: float, korean_terms: List[str]
    ) -> str:
        """기술 수준 결정"""
        if complexity > 0.7 or len(korean_terms) >= 2:
            return "고급"
        elif complexity > 0.4 or len(korean_terms) >= 1:
            return "중급"
        else:
            return "초급"

    def get_domain_specific_guidance(self, domain: str) -> Dict:
        """도메인별 지침 반환"""
        guidance = {
            "개인정보보호": {
                "key_laws": ["개인정보보호법", "정보통신망법"],
                "key_concepts": [
                    "정보주체",
                    "개인정보처리자",
                    "동의",
                    "목적외이용금지",
                    "만 14세 미만",
                    "법정대리인",
                ],
                "oversight_body": "개인정보보호위원회",
                "related_institutions": ["개인정보보호위원회", "개인정보침해신고센터"],
                "compliance_focus": "한국어 법령 용어 사용",
                "answer_patterns": [
                    "법적 근거 제시",
                    "기관명 정확 명시",
                    "절차 단계별 설명",
                ],
                "common_questions": [
                    "만 14세 미만 아동 동의",
                    "정책 수립 중요 요소",
                    "개인정보 관리체계",
                ],
            },
            "전자금융": {
                "key_laws": ["전자금융거래법", "전자서명법"],
                "key_concepts": [
                    "접근매체",
                    "전자서명",
                    "인증",
                    "분쟁조정",
                    "이용자",
                    "자료제출",
                ],
                "oversight_body": "금융감독원, 한국은행",
                "related_institutions": [
                    "전자금융분쟁조정위원회",
                    "금융감독원",
                    "한국은행",
                ],
                "compliance_focus": "한국어 금융 용어 사용",
                "answer_patterns": [
                    "분쟁조정 절차 설명",
                    "기관 역할 명시",
                    "법적 근거 제시",
                ],
                "common_questions": ["분쟁조정 신청 기관", "자료제출 요구 경우"],
            },
            "사이버보안": {
                "key_laws": ["정보통신망법", "개인정보보호법"],
                "key_concepts": [
                    "악성코드",
                    "침입탐지",
                    "보안관제",
                    "사고대응",
                    "트로이",
                    "RAT",
                    "SBOM",
                    "딥페이크",
                ],
                "oversight_body": "과학기술정보통신부, 경찰청",
                "related_institutions": ["한국인터넷진흥원", "사이버보안센터"],
                "compliance_focus": "한국어 보안 용어 사용",
                "answer_patterns": [
                    "탐지 지표 나열",
                    "대응 방안 제시",
                    "특징 상세 설명",
                ],
                "common_questions": [
                    "트로이 목마 특징",
                    "탐지 지표",
                    "SBOM 활용",
                    "딥페이크 대응",
                ],
            },
            "정보보안": {
                "key_laws": ["정보통신망법", "전자정부법"],
                "key_concepts": [
                    "정보보안관리체계",
                    "접근통제",
                    "암호화",
                    "백업",
                    "재해복구",
                ],
                "oversight_body": "과학기술정보통신부",
                "related_institutions": ["한국인터넷진흥원"],
                "compliance_focus": "한국어 기술 용어 사용",
                "answer_patterns": ["관리체계 설명", "보안조치 나열", "절차 단계 제시"],
                "common_questions": ["재해복구 계획", "관리체계 수립"],
            },
            "금융투자": {
                "key_laws": ["자본시장법", "금융투자업규정"],
                "key_concepts": [
                    "투자자보호",
                    "적합성원칙",
                    "설명의무",
                    "내부통제",
                    "금융투자업 구분",
                ],
                "oversight_body": "금융감독원, 금융위원회",
                "related_institutions": ["금융분쟁조정위원회", "금융감독원"],
                "compliance_focus": "한국어 투자 용어 사용",
                "answer_patterns": ["법령 근거 제시", "원칙 설명", "보호 방안 나열"],
                "common_questions": ["금융투자업 구분", "해당하지 않는 업무"],
            },
            "위험관리": {
                "key_laws": ["은행법", "보험업법", "자본시장법"],
                "key_concepts": [
                    "위험평가",
                    "내부통제",
                    "컴플라이언스",
                    "감사",
                    "위험 관리 계획",
                    "재해 복구",
                ],
                "oversight_body": "금융감독원",
                "related_institutions": ["금융감독원"],
                "compliance_focus": "한국어 관리 용어 사용",
                "answer_patterns": ["위험관리 절차", "평가 방법", "대응 체계"],
                "common_questions": [
                    "위험관리 요소",
                    "재해복구 계획",
                    "적절하지 않은 요소",
                ],
            },
        }

        return guidance.get(
            domain,
            {
                "key_laws": ["관련 법령"],
                "key_concepts": ["체계적 관리", "지속적 개선"],
                "oversight_body": "관계기관",
                "related_institutions": ["해당 전문기관"],
                "compliance_focus": "한국어 전용 답변",
                "answer_patterns": ["법령 근거", "관리 방안", "절차 설명"],
                "common_questions": [],
            },
        )

    def get_analysis_statistics(self) -> Dict:
        """분석 통계 반환"""
        return {
            "domain_frequency": dict(self.analysis_history["domain_frequency"]),
            "complexity_distribution": dict(
                self.analysis_history["complexity_distribution"]
            ),
            "compliance_check": dict(self.analysis_history["compliance_check"]),
            "intent_analysis_history": dict(
                self.analysis_history["intent_analysis_history"]
            ),
            "template_usage_stats": dict(self.analysis_history["template_usage_stats"]),
            "template_effectiveness": dict(
                self.analysis_history["template_effectiveness"]
            ),
            "mc_pattern_accuracy": dict(self.analysis_history["mc_pattern_accuracy"]),
            "institution_question_accuracy": dict(
                self.analysis_history["institution_question_accuracy"]
            ),
            "hint_provision_stats": dict(self.analysis_history["hint_provision_stats"]),
            "total_analyzed": len(self.analysis_history["question_patterns"]),
            "korean_terms_available": len(self.korean_financial_terms),
            "institutions_available": len(self.institution_database),
            "template_domains": len(self.korean_subjective_templates),
            "mc_patterns_available": len(self.mc_answer_patterns),
        }

    def validate_competition_compliance(self, answer: str, domain: str) -> Dict:
        """대회 규칙 준수 검증"""
        compliance = {
            "korean_only": True,
            "no_external_api": True,
            "appropriate_content": True,
            "technical_accuracy": True,
        }

        # 한국어 전용 확인
        import re

        english_chars = len(re.findall(r"[a-zA-Z]", answer))
        total_chars = len(re.sub(r"[^\w가-힣]", "", answer))

        if total_chars > 0:
            english_ratio = english_chars / total_chars
            compliance["korean_only"] = english_ratio < 0.1

        # 외부 의존성 확인
        external_indicators = ["http", "www", "api", "service", "cloud"]
        compliance["no_external_api"] = not any(
            indicator in answer.lower() for indicator in external_indicators
        )

        # 도메인 적절성 확인
        if domain in self.domain_keywords:
            domain_keywords = self.domain_keywords[domain]
            found_keywords = sum(
                1 for keyword in domain_keywords if keyword in answer.lower()
            )
            compliance["appropriate_content"] = found_keywords > 0

        return compliance

    def cleanup(self):
        """정리"""
        self._save_analysis_history()
