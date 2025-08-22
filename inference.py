# inference.py

"""
금융보안 AI 추론 시스템
- 문제 분류 및 처리
- 모델 추론 실행
- 결과 생성 및 저장
- 질문 의도 분석 및 답변 품질 검증
"""

import os
import time
import gc
import pandas as pd
from typing import Dict, List
from pathlib import Path

# 설정 파일 import
from config import (
    setup_environment,
    DEFAULT_MODEL_NAME,
    OPTIMIZATION_CONFIG,
    MEMORY_CONFIG,
    TIME_LIMITS,
    DEFAULT_FILES,
    FILE_VALIDATION,
)

# 현재 디렉토리 설정
current_dir = Path(__file__).parent.absolute()

# 로컬 모듈 import
from model_handler import SimpleModelHandler
from data_processor import SimpleDataProcessor
from knowledge_base import FinancialSecurityKnowledgeBase


class FinancialAIInference:
    """금융보안 AI 추론 시스템"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()

        # 환경 설정 초기화
        setup_environment()

        # 컴포넌트 초기화
        self.model_handler = SimpleModelHandler(verbose=verbose)
        self.data_processor = SimpleDataProcessor()
        self.knowledge_base = FinancialSecurityKnowledgeBase()

        # 성능 최적화 설정 로드
        self.optimization_config = OPTIMIZATION_CONFIG

    def process_single_question(self, question: str, question_id: str) -> str:
        """단일 질문 처리"""
        start_time = time.time()

        try:
            # 기본 분석
            question_type, max_choice = self.data_processor.extract_choice_range(question)
            domain = self.data_processor.extract_domain(question)
            difficulty = self.data_processor.analyze_question_difficulty(question)

            # 지식베이스 분석
            kb_analysis = self.knowledge_base.analyze_question(question)

            # 객관식 처리
            if question_type == "multiple_choice":
                answer = self._process_multiple_choice_with_enhanced_llm(
                    question, max_choice, domain, kb_analysis
                )
                return answer

            # 주관식 처리
            else:
                return self._process_subjective_with_enhanced_strategy(
                    question, question_id, domain, difficulty, kb_analysis, start_time
                )

        except Exception as e:
            if self.verbose:
                print(f"오류 발생: {e}")
            # 폴백 답변
            fallback = self._get_enhanced_intent_based_fallback(
                question, question_type, max_choice if "max_choice" in locals() else 5
            )
            return fallback

    def _process_subjective_with_enhanced_strategy(
        self, question: str, question_id: str, domain: str, difficulty: str, 
        kb_analysis: Dict, start_time: float
    ) -> str:
        """주관식 처리 강화 전략"""
        
        # 의도 분석
        intent_analysis = self.data_processor.analyze_question_intent(question)

        # 1차 시도: 기관 관련 질문 특별 처리
        if kb_analysis.get("institution_info", {}).get("is_institution_question", False):
            answer = self._process_institution_question_with_enhanced_llm(
                question, kb_analysis, intent_analysis
            )
            
            # 기관 질문 검증
            if self._validate_institution_answer(answer, question, kb_analysis):
                final_answer = self._enhanced_validate_and_improve_answer(
                    answer, question, "subjective", 5, domain, intent_analysis, kb_analysis
                )
                return final_answer

        # 2차 시도: 템플릿 기반 생성
        answer = self._process_subjective_with_template_examples(
            question, domain, intent_analysis, kb_analysis
        )

        # 답변 품질 검증
        if self._is_acceptable_answer(answer, question, intent_analysis):
            final_answer = self._enhanced_validate_and_improve_answer(
                answer, question, "subjective", 5, domain, intent_analysis, kb_analysis
            )
            return final_answer

        # 3차 시도: 다른 설정으로 재시도
        retry_answer = self._retry_subjective_generation(
            question, domain, intent_analysis, kb_analysis
        )

        if self._is_acceptable_answer(retry_answer, question, intent_analysis):
            final_answer = self._enhanced_validate_and_improve_answer(
                retry_answer, question, "subjective", 5, domain, intent_analysis, kb_analysis
            )
            return final_answer

        # 4차 시도: 의도 기반 폴백
        fallback_answer = self._get_intent_based_fallback_answer(
            question, intent_analysis, domain, kb_analysis
        )

        # 최종 처리
        final_answer = self._enhanced_validate_and_improve_answer(
            fallback_answer, question, "subjective", 5, domain, intent_analysis, kb_analysis
        )
        
        return final_answer

    def _validate_institution_answer(
        self, answer: str, question: str, kb_analysis: Dict
    ) -> bool:
        """기관 답변 검증"""
        if not answer or len(answer) < 10:
            return False
        
        # 기관 키워드 확인
        institution_keywords = ["위원회", "감독원", "은행", "기관", "센터"]
        has_institution = any(keyword in answer for keyword in institution_keywords)
        
        # 질문 맥락과 일치 확인
        if "전자금융" in question and "분쟁" in question:
            return "전자금융분쟁조정위원회" in answer or "금융감독원" in answer
        elif "개인정보" in question:
            return "개인정보보호위원회" in answer or "개인정보침해신고센터" in answer
        elif "한국은행" in question:
            return "한국은행" in answer
        
        return has_institution

    def _is_acceptable_answer(
        self, answer: str, question: str, intent_analysis: Dict = None
    ) -> bool:
        """답변 수용 가능성 검증"""
        if not answer:
            return False
        
        # 기본 길이 검증
        if len(answer) < 15:
            return False
        
        # 치명적인 반복 패턴만 확인
        if self.model_handler.detect_critical_repetitive_patterns(answer):
            return False
        
        # 한국어 비율 검증
        korean_ratio = self.data_processor.calculate_korean_ratio(answer)
        if korean_ratio < 0.5:
            return False
        
        # 의미 있는 내용 확인
        meaningful_keywords = [
            "법령", "규정", "조치", "관리", "보안", "방안", "절차", "기준", 
            "정책", "체계", "시스템", "통제", "특징", "지표", "탐지", "대응",
            "기관", "위원회", "감독원"
        ]
        if not any(word in answer for word in meaningful_keywords):
            return False
        
        # 의도별 특별 검증
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            if answer_type == "기관명":
                institution_keywords = ["위원회", "감독원", "은행", "기관", "센터"]
                if not any(keyword in answer for keyword in institution_keywords):
                    return False
        
        return True

    def _retry_subjective_generation(
        self, question: str, domain: str, intent_analysis: Dict, kb_analysis: Dict
    ) -> str:
        """주관식 재시도 생성"""
        
        # 다른 템플릿 예시 사용
        alternative_hints = {"retry_mode": True, "domain": domain}
        
        # 추가 템플릿 예시 수집
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            intent_key = self._map_intent_to_key(primary_intent)
            
            # 다른 도메인의 템플릿도 참고
            alternative_domains = ["사이버보안", "개인정보보호", "전자금융", "정보보안"]
            for alt_domain in alternative_domains:
                if alt_domain != domain:
                    alt_templates = self.knowledge_base.get_template_examples(alt_domain, intent_key)
                    if alt_templates:
                        alternative_hints["alternative_templates"] = alt_templates[:2]
                        break
        
        # 기관 정보가 있으면 추가
        if kb_analysis.get("institution_info", {}).get("is_institution_question", False):
            institution_type = kb_analysis["institution_info"].get("institution_type")
            if institution_type:
                alternative_hints["institution_hints"] = (
                    self.knowledge_base.get_institution_hints(institution_type)
                )
        
        return self.model_handler.generate_answer(
            question, "subjective", 5, intent_analysis, alternative_hints
        )

    def _get_intent_based_fallback_answer(
        self, question: str, intent_analysis: Dict, domain: str, kb_analysis: Dict
    ) -> str:
        """의도 기반 폴백 답변"""
        
        if not intent_analysis:
            return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."
        
        primary_intent = intent_analysis.get("primary_intent", "일반")
        answer_type = intent_analysis.get("answer_type_required", "설명형")
        
        # 의도별 맞춤 폴백
        fallback_templates = {
            "기관_묻기": self._get_institution_fallback(question, domain),
            "특징_묻기": self._get_feature_fallback(question, domain),
            "지표_묻기": self._get_indicator_fallback(question, domain),
            "방안_묻기": self._get_solution_fallback(question, domain),
            "절차_묻기": self._get_procedure_fallback(question, domain),
            "조치_묻기": self._get_measure_fallback(question, domain),
        }
        
        intent_key = self._map_intent_to_key(primary_intent)
        if intent_key in fallback_templates:
            return fallback_templates[intent_key]
        
        # 도메인별 기본 폴백
        domain_fallbacks = {
            "사이버보안": "사이버보안 위협에 대응하기 위해 다층 방어체계를 구축하고 실시간 모니터링을 수행해야 합니다.",
            "개인정보보호": "개인정보보호법에 따라 정보주체의 권리를 보장하고 적절한 보호조치를 이행해야 합니다.",
            "전자금융": "전자금융거래법에 따라 안전한 거래환경을 제공하고 이용자 보호를 위한 조치를 시행해야 합니다.",
            "정보보안": "정보보안관리체계를 구축하고 보안정책에 따라 체계적인 관리를 수행해야 합니다.",
            "금융투자": "자본시장법에 따라 투자자 보호와 시장 공정성 확보를 위한 조치를 시행해야 합니다.",
            "위험관리": "위험관리 체계를 구축하고 체계적인 위험평가와 대응방안을 수립해야 합니다.",
        }
        
        return domain_fallbacks.get(domain, "관련 법령과 규정에 따라 체계적인 관리가 필요합니다.")

    def _get_institution_fallback(self, question: str, domain: str) -> str:
        """기관 관련 폴백"""
        if "전자금융" in question and "분쟁" in question:
            return "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다."
        elif "개인정보" in question:
            return "개인정보보호위원회에서 개인정보 보호에 관한 업무를 총괄하고 있습니다."
        elif "한국은행" in question:
            return "한국은행에서 통화신용정책 수행과 지급결제제도 운영을 담당합니다."
        else:
            return "관련 전문 기관에서 해당 업무를 담당하고 있습니다."

    def _get_feature_fallback(self, question: str, domain: str) -> str:
        """특징 관련 폴백"""
        if "트로이" in question or "악성코드" in question:
            return "트로이 목마는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다."
        elif domain == "사이버보안":
            return "해당 보안 위협의 주요 특징을 체계적으로 분석하여 대응 방안을 수립해야 합니다."
        else:
            return "주요 특징을 체계적으로 분석하고 관련 법령에 따라 관리해야 합니다."

    def _get_indicator_fallback(self, question: str, domain: str) -> str:
        """지표 관련 폴백"""
        if "트로이" in question or "악성코드" in question:
            return "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴과 시스템 동작 분석에서 비인가 프로세스 실행이 주요 탐지 지표입니다."
        elif domain == "사이버보안":
            return "주요 탐지 지표를 통해 실시간 모니터링과 이상 징후 분석을 수행해야 합니다."
        else:
            return "관련 지표를 체계적으로 분석하고 모니터링하여 적절한 대응을 수행해야 합니다."

    def _get_solution_fallback(self, question: str, domain: str) -> str:
        """방안 관련 폴백"""
        if "딥페이크" in question:
            return "딥페이크 기술 악용에 대비하여 다층 방어체계 구축과 실시간 탐지 시스템 도입 등의 종합적 대응방안이 필요합니다."
        elif domain == "사이버보안":
            return "다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영하는 등의 종합적 보안 강화 방안이 필요합니다."
        else:
            return "체계적인 관리 방안을 수립하고 관련 법령과 규정에 따라 지속적인 개선을 수행해야 합니다."

    def _get_procedure_fallback(self, question: str, domain: str) -> str:
        """절차 관련 폴백"""
        return "관련 절차에 따라 단계별로 체계적인 수행과 지속적인 관리가 필요합니다."

    def _get_measure_fallback(self, question: str, domain: str) -> str:
        """조치 관련 폴백"""
        return "적절한 보안 조치를 시행하고 관련 법령과 규정에 따라 지속적인 관리가 필요합니다."

    def _map_intent_to_key(self, primary_intent: str) -> str:
        """의도를 키로 매핑"""
        if "기관" in primary_intent:
            return "기관_묻기"
        elif "특징" in primary_intent:
            return "특징_묻기"
        elif "지표" in primary_intent:
            return "지표_묻기"
        elif "방안" in primary_intent:
            return "방안_묻기"
        elif "절차" in primary_intent:
            return "절차_묻기"
        elif "조치" in primary_intent:
            return "조치_묻기"
        else:
            return "일반"

    def _process_multiple_choice_with_enhanced_llm(
        self, question: str, max_choice: int, domain: str, kb_analysis: Dict
    ) -> str:
        """객관식 처리"""

        # 지식베이스에서 힌트 정보 수집
        pattern_hints = None
        if self.optimization_config["mc_pattern_priority"]:
            pattern_hints = self.knowledge_base.get_mc_pattern_hints(question)

        # LLM 답변 생성
        answer = self.model_handler.generate_answer(
            question,
            "multiple_choice",
            max_choice,
            intent_analysis=None,
            domain_hints={"domain": domain, "pattern_hints": pattern_hints},
        )

        # 답변 범위 검증
        if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
            return answer
        else:
            # 범위 오류 시 재시도
            fallback = self._enhanced_retry_mc_with_llm(question, max_choice, domain)
            return fallback

    def _process_subjective_with_template_examples(
        self, question: str, domain: str, intent_analysis: Dict, kb_analysis: Dict
    ) -> str:
        """템플릿 예시를 활용한 주관식 처리"""

        # 템플릿 예시 정보 수집
        template_examples = None
        if (
            intent_analysis
            and intent_analysis.get("intent_confidence", 0)
            >= self.optimization_config["intent_confidence_threshold"]
        ):

            primary_intent = intent_analysis.get("primary_intent", "일반")

            # 의도별 특화 템플릿 예시 수집
            if self.optimization_config["template_preference"]:
                intent_key = self._map_intent_to_key(primary_intent)

                # 템플릿 예시 수집
                template_examples = self.knowledge_base.get_template_examples(domain, intent_key)

        # LLM 답변 생성
        answer = self.model_handler.generate_answer(
            question,
            "subjective",
            5,
            intent_analysis,
            domain_hints={
                "domain": domain,
                "template_examples": template_examples,
                "template_guidance": True,
            },
        )

        return answer

    def _enhanced_retry_mc_with_llm(self, question: str, max_choice: int, domain: str) -> str:
        """객관식 LLM 재시도"""
        # 컨텍스트 기반 재시도
        context_hints = self.model_handler._analyze_mc_context(question, domain)
        retry_answer = self.model_handler.generate_answer(
            question,
            "multiple_choice",
            max_choice,
            intent_analysis=None,
            domain_hints={
                "domain": domain,
                "context_hints": context_hints,
                "retry_mode": True,
            },
        )

        # 범위 벗어나면 컨텍스트 기반 재요청
        if not (retry_answer and retry_answer.isdigit() and 1 <= int(retry_answer) <= max_choice):
            retry_answer = self.model_handler.generate_contextual_mc_answer(
                question, max_choice, domain
            )

        # 최종 실패시 유효한 답변 강제 생성
        if not (retry_answer and retry_answer.isdigit() and 1 <= int(retry_answer) <= max_choice):
            retry_answer = str((max_choice + 1) // 2)

        return retry_answer

    def _process_institution_question_with_enhanced_llm(
        self, question: str, kb_analysis: Dict, intent_analysis: Dict
    ) -> str:
        """기관 질문 처리"""
        institution_info = kb_analysis.get("institution_info", {})

        # 기관 정보를 힌트로 제공
        institution_hints = None
        if institution_info.get("is_institution_question", False):
            institution_type = institution_info.get("institution_type")
            if institution_type and institution_info.get("confidence", 0) > 0.5:
                # 지식베이스에서 힌트 정보 가져오기
                institution_hints = self.knowledge_base.get_institution_hints(institution_type)

        # LLM 답변 생성
        answer = self.model_handler.generate_answer(
            question,
            "subjective",
            5,
            intent_analysis,
            domain_hints={"institution_hints": institution_hints},
        )

        # 기관 답변 검증 및 개선
        answer = self._enhance_institution_answer(answer, question, institution_info)

        return answer

    def _enhance_institution_answer(self, answer: str, question: str, institution_info: Dict) -> str:
        """기관 답변 개선"""
        if not answer:
            return answer

        # 기관명이 포함되지 않은 경우 추가
        institution_keywords = ["위원회", "감독원", "은행", "기관", "센터"]
        has_institution = any(keyword in answer for keyword in institution_keywords)

        if not has_institution:
            # 질문 내용을 바탕으로 적절한 기관명 추가
            if "전자금융" in question and "분쟁" in question:
                answer = "전자금융분쟁조정위원회에서 " + answer
            elif "개인정보" in question:
                answer = "개인정보보호위원회에서 " + answer
            elif "한국은행" in question or "자료제출" in question:
                answer = "한국은행에서 " + answer
            elif "금융투자" in question and "분쟁" in question:
                answer = "금융분쟁조정위원회에서 " + answer

        return answer

    def _enhanced_validate_and_improve_answer(
        self,
        answer: str,
        question: str,
        question_type: str,
        max_choice: int,
        domain: str,
        intent_analysis: Dict = None,
        kb_analysis: Dict = None,
    ) -> str:
        """답변 검증 및 개선"""

        if question_type == "multiple_choice":
            return answer

        # 주관식 품질 검증 및 개선
        original_answer = answer

        # 텍스트 복구 및 정리
        recovered_answer = self.data_processor.clean_korean_text(answer)
        if recovered_answer != answer:
            answer = recovered_answer

        # 기본 유효성 검증
        is_valid = self.data_processor.validate_korean_answer(
            answer, question_type, max_choice, question
        )

        if not is_valid:
            # 검증 실패시 개선된 폴백 사용
            if intent_analysis:
                answer = self._get_intent_based_fallback_answer(
                    question, intent_analysis, domain, kb_analysis
                )
            else:
                answer = "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        # 한국어 비율 검증 및 개선
        korean_ratio = self.data_processor.calculate_korean_ratio(answer)
        if korean_ratio < 0.6:
            if intent_analysis:
                answer = self._get_intent_based_fallback_answer(
                    question, intent_analysis, domain, kb_analysis
                )
            else:
                answer = "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        # 의도 일치성 검증 및 개선
        if intent_analysis:
            intent_match = self.data_processor.validate_answer_intent_match(
                answer, question, intent_analysis
            )
            if not intent_match:
                # 의도 불일치시 맞춤형 재생성
                answer = self._get_intent_based_fallback_answer(
                    question, intent_analysis, domain, kb_analysis
                )
                # 재검증
                intent_match_retry = self.data_processor.validate_answer_intent_match(
                    answer, question, intent_analysis
                )

        # 답변 품질 평가 및 개선
        quality_score = self._calculate_enhanced_quality_score(answer, question, intent_analysis)
        if quality_score < 0.5:
            if intent_analysis:
                improved_answer = self._get_intent_based_fallback_answer(
                    question, intent_analysis, domain, kb_analysis
                )
            else:
                improved_answer = "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."
                
            improved_quality = self._calculate_enhanced_quality_score(
                improved_answer, question, intent_analysis
            )

            if improved_quality > quality_score:
                answer = improved_answer

        # 문법 및 구조 개선
        grammar_improved_answer = self.data_processor.fix_grammatical_structure(answer)
        if grammar_improved_answer != answer:
            answer = grammar_improved_answer

        # 답변 구조 최적화
        structure_improved_answer = self._optimize_answer_structure(answer, intent_analysis)
        if structure_improved_answer != answer:
            answer = structure_improved_answer

        # 길이 최적화
        answer = self._optimize_answer_length(answer)

        # 최종 정규화
        answer = self.data_processor.normalize_korean_answer(answer, question_type, max_choice)

        return answer

    def _optimize_answer_structure(self, answer: str, intent_analysis: Dict = None) -> str:
        """답변 구조 최적화"""
        if not answer or len(answer) < 20:
            return answer

        # 의도별 구조 개선
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")

            # 기관명 답변 구조 개선
            if answer_type == "기관명":
                if not any(word in answer for word in ["위원회", "감독원", "은행", "기관"]):
                    if "분쟁조정" in answer:
                        answer = "전자금융분쟁조정위원회에서 " + answer
                    elif "개인정보" in answer:
                        answer = "개인정보보호위원회에서 " + answer

            # 특징 설명 구조 개선
            elif answer_type == "특징설명":
                if not answer.startswith(("주요 특징", "특징", "특성")):
                    answer = "주요 특징은 " + answer

            # 지표 나열 구조 개선
            elif answer_type == "지표나열":
                if not any(word in answer[:50] for word in ["지표", "탐지", "징후"]):
                    answer = "주요 탐지 지표는 " + answer

            # 방안 제시 구조 개선
            elif answer_type == "방안제시":
                if not any(word in answer[:50] for word in ["방안", "대책", "조치"]):
                    answer = "주요 대응 방안은 " + answer

        # 문장 연결 개선
        sentences = answer.split(". ")
        if len(sentences) > 1:
            improved_sentences = []
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) < 5:
                    continue

                # 접속어 추가
                if i > 0 and len(sentence) > 10:
                    if not any(
                        sentence.startswith(word)
                        for word in [
                            "또한",
                            "그리고",
                            "이를",
                            "따라서",
                            "그러므로",
                            "하지만",
                            "그러나",
                        ]
                    ):
                        if "방안" in sentence or "조치" in sentence:
                            sentence = "또한 " + sentence
                        elif "법령" in sentence or "규정" in sentence:
                            sentence = "이를 위해 " + sentence

                improved_sentences.append(sentence)

            answer = ". ".join(improved_sentences)

        # 마침표 정리
        if answer and not answer.endswith("."):
            answer += "."

        return answer

    def _get_enhanced_intent_based_fallback(
        self, question: str, question_type: str, max_choice: int
    ) -> str:
        """의도 기반 폴백 답변"""
        
        # 질문 의도 분석
        intent_analysis = self.data_processor.analyze_question_intent(question)
        domain = self.data_processor.extract_domain(question)
        
        if question_type == "multiple_choice":
            return self._get_enhanced_safe_mc_answer_with_llm(question, max_choice, domain)
        else:
            return self._get_intent_based_fallback_answer(
                question, intent_analysis, domain, {}
            )

    def _calculate_enhanced_quality_score(
        self, answer: str, question: str, intent_analysis: Dict = None
    ) -> float:
        """품질 점수 계산"""
        if not answer:
            return 0.0

        score = 0.0

        # 반복 패턴 페널티
        if self.model_handler.detect_critical_repetitive_patterns(answer):
            return 0.1

        # 한국어 비율
        korean_ratio = self.data_processor.calculate_korean_ratio(answer)
        score += korean_ratio * 0.2

        # 텍스트 복구 품질
        has_broken_chars = any(char in answer for char in ["ト", "リ", "ス", "ン", "윋", "젂", "엯"])
        if not has_broken_chars:
            score += 0.15

        # 길이 적절성
        length = len(answer)
        if 30 <= length <= 500:
            score += 0.2
        elif 20 <= length < 30 or 500 < length <= 600:
            score += 0.15
        elif 15 <= length < 20:
            score += 0.1

        # 문장 구조
        if answer.endswith((".", "다", "요", "함")):
            score += 0.15

        sentences = answer.split(".")
        if len(sentences) >= 2:
            score += 0.1

        # 전문성
        domain_keywords = self.model_handler._get_domain_keywords(question)
        found_keywords = sum(1 for keyword in domain_keywords if keyword in answer)
        if found_keywords > 0:
            score += min(found_keywords / len(domain_keywords), 1.0) * 0.15

        # 의도 일치성
        if intent_analysis:
            intent_match = self.data_processor.validate_answer_intent_match(
                answer, question, intent_analysis
            )
            if intent_match:
                score += 0.25
            else:
                score += 0.15
        else:
            score += 0.2

        return min(score, 1.0)

    def _optimize_answer_length(self, answer: str) -> str:
        """답변 길이 최적화"""
        if not answer:
            return answer

        # 너무 긴 답변 축약
        if len(answer) > 500:
            sentences = answer.split(". ")
            if len(sentences) > 4:
                answer = ". ".join(sentences[:4])
                if not answer.endswith("."):
                    answer += "."

        # 너무 짧은 답변 보강
        elif len(answer) < 30:
            if not answer.endswith("."):
                answer += "."
            if "법령" not in answer and "규정" not in answer and len(answer) < 40:
                answer += " 관련 법령과 규정을 준수하여 체계적으로 관리해야 합니다."

        return answer

    def _get_enhanced_safe_mc_answer_with_llm(
        self, question: str, max_choice: int, domain: str = "일반"
    ) -> str:
        """안전한 객관식 답변 생성"""
        if max_choice <= 0:
            max_choice = 5

        # LLM을 통한 안전한 답변 생성
        fallback_answer = self.model_handler.generate_fallback_mc_answer(question, max_choice, domain)

        # LLM 결과가 유효하지 않은 경우에만 최후 수단 사용
        if not (
            fallback_answer
            and fallback_answer.isdigit()
            and 1 <= int(fallback_answer) <= max_choice
        ):
            import random
            fallback_answer = str(random.randint(1, max_choice))

        return fallback_answer

    def _simple_save_csv(self, df: pd.DataFrame, filepath: str) -> bool:
        """CSV 저장"""
        filepath = Path(filepath)

        try:
            df.to_csv(filepath, index=False, encoding=FILE_VALIDATION["encoding"])
            if self.verbose:
                print(f"\n결과 저장 완료: {filepath}")
            return True

        except PermissionError as e:
            if self.verbose:
                print(f"\n파일 저장 권한 오류: {e}")
                print("파일이 다른 프로그램에서 열려있는지 확인하세요.")
            return False

        except Exception as e:
            if self.verbose:
                print(f"\n파일 저장 중 오류: {e}")
            return False

    def execute_inference(
        self,
        test_file: str = None,
        submission_file: str = None,
        output_file: str = None,
    ) -> Dict:
        """전체 추론 실행"""

        # 기본 파일 경로 사용
        test_file = test_file or DEFAULT_FILES["test_file"]
        submission_file = submission_file or DEFAULT_FILES["submission_file"]
        output_file = output_file or DEFAULT_FILES["output_file"]

        # 데이터 로드
        try:
            test_df = pd.read_csv(test_file)
            submission_df = pd.read_csv(submission_file)
        except Exception as e:
            raise RuntimeError(f"데이터 로드 실패: {e}")

        return self.execute_inference_with_data(test_df, submission_df, output_file)

    def execute_inference_with_data(
        self,
        test_df: pd.DataFrame,
        submission_df: pd.DataFrame,
        output_file: str = None,
    ) -> Dict:
        """데이터프레임으로 추론 실행"""

        output_file = output_file or DEFAULT_FILES["output_file"]

        print(f"데이터 로드 완료: {len(test_df)}개 문항")

        answers = []
        total_questions = len(test_df)
        inference_start_time = time.time()

        # enumerate를 사용하여 0부터 시작하는 정확한 인덱스 사용
        for question_idx, (original_idx, row) in enumerate(test_df.iterrows()):
            question = row["Question"]
            question_id = row["ID"]

            # 추론 수행
            answer = self.process_single_question(question, question_id)
            answers.append(answer)

            # 메모리 관리
            if (question_idx + 1) % MEMORY_CONFIG["gc_frequency"] == 0:
                gc.collect()

        # 결과 저장
        submission_df["Answer"] = answers
        save_success = self._simple_save_csv(submission_df, output_file)

        if not save_success:
            print(f"지정된 파일로 저장에 실패했습니다: {output_file}")
            print("파일이 다른 프로그램에서 열려있거나 권한 문제일 수 있습니다.")

        return self._get_results_summary()

    def _get_results_summary(self) -> Dict:
        """결과 요약"""
        return {
            "success": True,
            "total_time": time.time() - self.start_time,
        }

    def cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, "model_handler"):
                self.model_handler.cleanup()

            if hasattr(self, "data_processor"):
                self.data_processor.cleanup()

            if hasattr(self, "knowledge_base"):
                self.knowledge_base.cleanup()

            gc.collect()

        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")


def main():
    """메인 함수"""

    engine = None
    try:
        # AI 엔진 초기화
        engine = FinancialAIInference(verbose=True)

        # 추론 실행
        results = engine.execute_inference()

        if results["success"]:
            print("\n추론 완료")
            print(f"총 처리시간: {results['total_time']:.1f}초")

    except KeyboardInterrupt:
        print("\n추론 중단됨")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.cleanup()


if __name__ == "__main__":
    main()