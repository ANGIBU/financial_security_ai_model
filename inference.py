# inference.py

"""
금융보안 AI 추론 시스템
- 문제 분류 및 처리
- 모델 추론 실행
- 결과 생성 및 저장
- 학습 데이터 관리
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
    PROGRESS_CONFIG,
    DEFAULT_FILES,
    STATS_CONFIG,
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

        if verbose:
            print("추론 시스템 초기화")

        # 컴포넌트 초기화
        if verbose:
            print("1/3 모델 핸들러 초기화...")
        self.model_handler = SimpleModelHandler(verbose=verbose)

        if verbose:
            print("2/3 데이터 프로세서 초기화...")
        self.data_processor = SimpleDataProcessor()

        if verbose:
            print("3/3 지식베이스 초기화...")
        self.knowledge_base = FinancialSecurityKnowledgeBase()

        # 통계 데이터
        self.stats = {
            "total": 0,
            "mc_count": 0,
            "subj_count": 0,
            "model_success": 0,
            "korean_compliance": 0,
            "processing_times": [],
            "domain_stats": {},
            "difficulty_stats": {"초급": 0, "중급": 0, "고급": 0},
            "quality_scores": [],
            "mc_answers_by_range": {
                3: {"1": 0, "2": 0, "3": 0},
                4: {"1": 0, "2": 0, "3": 0, "4": 0},
                5: {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
            },
            "choice_range_errors": 0,
            "validation_errors": 0,
            "intent_analysis_accuracy": 0,
            "intent_match_success": 0,
            "institution_questions": 0,
            "template_usage": 0,
            "answer_quality_by_intent": {},
            "mc_context_accuracy": 0,
            "mc_pattern_matches": 0,
            "high_confidence_intent": 0,
            "intent_specific_answers": 0,
            "quality_improvement": 0,
            "fallback_avoidance": 0,
            "domain_intent_match": {},
            "answer_length_optimization": 0,
            "korean_enhancement": 0,
            "template_effectiveness": {},
            "mc_domain_accuracy": {},
            "institution_answer_accuracy": 0,
            "negative_positive_balance": {"negative": 0, "positive": 0, "neutral": 0},
            "llm_usage_rate": 0,
            "hint_usage_rate": 0,
            "text_recovery_count": 0,
            "grammar_fix_count": 0,
            "korean_ratio_improvements": 0,
            "answer_structure_improvements": 0,
            "template_examples_used": 0,
            "llm_generation_count": 0,
            "template_guided_answers": 0,
        }

        # 성능 최적화 설정 (config.py에서 로드)
        self.optimization_config = OPTIMIZATION_CONFIG

        if verbose:
            print("초기화 완료")

    def process_single_question(self, question: str, question_id: str) -> str:
        """단일 질문 처리"""
        start_time = time.time()

        try:
            # 기본 분석
            question_type, max_choice = self.data_processor.extract_choice_range(
                question
            )
            domain = self.data_processor.extract_domain(question)
            difficulty = self.data_processor.analyze_question_difficulty(question)

            # 지식베이스 분석
            kb_analysis = self.knowledge_base.analyze_question(question)

            # 객관식 우선 처리
            if question_type == "multiple_choice":
                answer = self._process_multiple_choice_with_enhanced_llm(
                    question, max_choice, domain, kb_analysis
                )
                self._update_mc_stats(
                    question_type,
                    domain,
                    difficulty,
                    time.time() - start_time,
                    answer,
                    max_choice,
                )
                return answer

            # 주관식 처리 - 템플릿 예시 활용 강화
            else:
                intent_analysis = self.data_processor.analyze_question_intent(question)
                self.stats["intent_analysis_accuracy"] += 1

                # 신뢰도 확인
                if (
                    intent_analysis.get("intent_confidence", 0)
                    >= self.optimization_config["intent_confidence_threshold"]
                ):
                    self.stats["high_confidence_intent"] += 1

                # 기관 관련 질문 우선 처리
                if kb_analysis.get("institution_info", {}).get(
                    "is_institution_question", False
                ):
                    self.stats["institution_questions"] += 1
                    answer = self._process_institution_question_with_enhanced_llm(
                        question, kb_analysis, intent_analysis
                    )
                else:
                    # 일반 주관식 처리 - 템플릿 예시 활용
                    answer = self._process_subjective_with_template_examples(
                        question, domain, intent_analysis, kb_analysis
                    )

                # 강화된 품질 검증 및 개선
                final_answer = self._enhanced_validate_and_improve_answer(
                    answer,
                    question,
                    question_type,
                    max_choice,
                    domain,
                    intent_analysis,
                    kb_analysis,
                )

                # 통계 업데이트
                self._update_subj_stats(
                    question_type,
                    domain,
                    difficulty,
                    time.time() - start_time,
                    intent_analysis,
                    final_answer,
                )

                return final_answer

        except Exception as e:
            if self.verbose:
                print(f"오류 발생: {e}")
            # 안전한 폴백 답변
            fallback = self._get_enhanced_safe_fallback_with_llm(
                question, question_type, max_choice if "max_choice" in locals() else 5
            )
            self._update_stats(
                question_type if "question_type" in locals() else "multiple_choice",
                domain if "domain" in locals() else "일반",
                difficulty if "difficulty" in locals() else "초급",
                time.time() - start_time,
            )
            return fallback

    def _process_multiple_choice_with_enhanced_llm(
        self, question: str, max_choice: int, domain: str, kb_analysis: Dict
    ) -> str:
        """강화된 객관식 처리"""

        # 지식베이스에서 힌트 정보 수집
        pattern_hints = None
        if self.optimization_config["mc_pattern_priority"]:
            pattern_hints = self.knowledge_base.get_mc_pattern_hints(question)
            if pattern_hints:
                self.stats["mc_pattern_matches"] += 1
                self.stats["hint_usage_rate"] += 1

        # 강화된 LLM 답변 생성
        answer = self.model_handler.generate_answer(
            question,
            "multiple_choice",
            max_choice,
            intent_analysis=None,
            domain_hints={"domain": domain, "pattern_hints": pattern_hints},
        )

        # 답변 범위 검증 및 복구
        if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
            # 선택지 분포 업데이트
            if max_choice in self.stats["mc_answers_by_range"]:
                self.stats["mc_answers_by_range"][max_choice][answer] += 1

            # 도메인별 정확도 추적
            if domain not in self.stats["mc_domain_accuracy"]:
                self.stats["mc_domain_accuracy"][domain] = {"total": 0, "success": 0}
            self.stats["mc_domain_accuracy"][domain]["total"] += 1
            self.stats["mc_domain_accuracy"][domain]["success"] += 1

            self.stats["model_success"] += 1
            self.stats["korean_compliance"] += 1
            self.stats["llm_usage_rate"] += 1
            self.stats["llm_generation_count"] += 1
            return answer
        else:
            # 범위 오류 시 강화된 LLM 재시도
            self.stats["choice_range_errors"] += 1
            fallback = self._enhanced_retry_mc_with_llm(question, max_choice, domain)

            # 도메인별 정확도 추적
            if domain not in self.stats["mc_domain_accuracy"]:
                self.stats["mc_domain_accuracy"][domain] = {"total": 0, "success": 0}
            self.stats["mc_domain_accuracy"][domain]["total"] += 1

            if max_choice in self.stats["mc_answers_by_range"]:
                self.stats["mc_answers_by_range"][max_choice][fallback] += 1

            self.stats["llm_usage_rate"] += 1
            self.stats["llm_generation_count"] += 1
            return fallback

    def _process_subjective_with_template_examples(
        self, question: str, domain: str, intent_analysis: Dict, kb_analysis: Dict
    ) -> str:
        """템플릿 예시를 활용한 강화된 주관식 처리"""

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
                if "기관" in primary_intent:
                    intent_key = "기관_묻기"
                elif "특징" in primary_intent:
                    intent_key = "특징_묻기"
                elif "지표" in primary_intent:
                    intent_key = "지표_묻기"
                elif "방안" in primary_intent:
                    intent_key = "방안_묻기"
                elif "절차" in primary_intent:
                    intent_key = "절차_묻기"
                elif "조치" in primary_intent:
                    intent_key = "조치_묻기"
                else:
                    intent_key = "일반"

                # 템플릿 예시 수집 - 실제 예시 텍스트 반환
                template_examples = self.knowledge_base.get_template_examples(
                    domain, intent_key
                )
                if template_examples:
                    self.stats["intent_specific_answers"] += 1
                    self.stats["template_usage"] += 1
                    self.stats["hint_usage_rate"] += 1
                    self.stats["template_examples_used"] += 1
                    self.stats["template_guided_answers"] += 1

        # 강화된 LLM 답변 생성 - 템플릿 예시 전달
        answer = self.model_handler.generate_answer(
            question,
            "subjective",
            5,
            intent_analysis,
            domain_hints={
                "domain": domain, 
                "template_examples": template_examples,
                "template_guidance": True
            },
        )

        self.stats["llm_usage_rate"] += 1
        self.stats["llm_generation_count"] += 1
        return answer

    def _enhanced_retry_mc_with_llm(
        self, question: str, max_choice: int, domain: str
    ) -> str:
        """강화된 객관식 LLM 재시도"""
        # 더 명확한 프롬프트로 LLM 재시도
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

        # 여전히 범위 밖이면 컨텍스트 기반으로 LLM에게 다시 요청
        if not (
            retry_answer
            and retry_answer.isdigit()
            and 1 <= int(retry_answer) <= max_choice
        ):
            retry_answer = self.model_handler.generate_contextual_mc_answer(
                question, max_choice, domain
            )

        # 최종적으로도 실패하면 유효한 답변 강제 생성
        if not (
            retry_answer
            and retry_answer.isdigit()
            and 1 <= int(retry_answer) <= max_choice
        ):
            retry_answer = str((max_choice + 1) // 2)

        self.stats["llm_generation_count"] += 1
        return retry_answer

    def _process_institution_question_with_enhanced_llm(
        self, question: str, kb_analysis: Dict, intent_analysis: Dict
    ) -> str:
        """강화된 기관 질문 처리"""
        institution_info = kb_analysis.get("institution_info", {})

        # 기관 정보를 힌트로 제공하여 LLM이 답변 생성
        institution_hints = None
        if institution_info.get("is_institution_question", False):
            institution_type = institution_info.get("institution_type")
            if institution_type and institution_info.get("confidence", 0) > 0.5:
                # 신뢰도 높은 기관 질문 - 지식베이스에서 힌트 정보 가져오기
                institution_hints = self.knowledge_base.get_institution_hints(
                    institution_type
                )
                self.stats["intent_specific_answers"] += 1
                self.stats["institution_answer_accuracy"] += 1
                self.stats["hint_usage_rate"] += 1

        # 강화된 LLM 답변 생성
        answer = self.model_handler.generate_answer(
            question,
            "subjective",
            5,
            intent_analysis,
            domain_hints={"institution_hints": institution_hints},
        )

        # 기관 답변 검증 및 개선
        answer = self._enhance_institution_answer(answer, question, institution_info)

        self.stats["llm_usage_rate"] += 1
        self.stats["llm_generation_count"] += 1
        return answer

    def _enhance_institution_answer(
        self, answer: str, question: str, institution_info: Dict
    ) -> str:
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
        """강화된 답변 검증 및 개선"""

        if question_type == "multiple_choice":
            return answer

        # 주관식 품질 검증 및 개선
        original_answer = answer
        improvement_count = 0

        # 1단계: 텍스트 복구 및 정리
        recovered_answer = self.data_processor.clean_korean_text(answer)
        if recovered_answer != answer:
            answer = recovered_answer
            improvement_count += 1
            self.stats["text_recovery_count"] += 1

        # 2단계: 기본 유효성 검증
        is_valid = self.data_processor.validate_korean_answer(
            answer, question_type, max_choice, question
        )

        if not is_valid:
            self.stats["validation_errors"] += 1
            answer = self._get_enhanced_improved_answer_with_llm(
                question, domain, intent_analysis, kb_analysis, "validation_failed"
            )
            improvement_count += 1

        # 3단계: 한국어 비율 검증 및 개선
        korean_ratio = self.data_processor.calculate_korean_ratio(answer)
        if korean_ratio < self.optimization_config["korean_ratio_threshold"]:
            answer = self._get_enhanced_improved_answer_with_llm(
                question, domain, intent_analysis, kb_analysis, "korean_ratio_low"
            )
            improvement_count += 1
            self.stats["korean_enhancement"] += 1
            self.stats["korean_ratio_improvements"] += 1

        # 4단계: 의도 일치성 검증 및 개선
        if intent_analysis:
            intent_match = self.data_processor.validate_answer_intent_match(
                answer, question, intent_analysis
            )
            if intent_match:
                self.stats["intent_match_success"] += 1
            else:
                # 의도 불일치시 템플릿 가이드와 함께 LLM 재생성
                answer = self._get_enhanced_improved_answer_with_llm(
                    question, domain, intent_analysis, kb_analysis, "intent_mismatch"
                )
                improvement_count += 1
                # 재검증
                intent_match_retry = self.data_processor.validate_answer_intent_match(
                    answer, question, intent_analysis
                )
                if intent_match_retry:
                    self.stats["intent_match_success"] += 1

        # 5단계: 답변 품질 평가 및 개선
        quality_score = self._calculate_enhanced_quality_score(
            answer, question, intent_analysis
        )
        if quality_score < self.optimization_config["quality_threshold"]:
            improved_answer = self._get_enhanced_improved_answer_with_llm(
                question, domain, intent_analysis, kb_analysis, "quality_low"
            )
            improved_quality = self._calculate_enhanced_quality_score(
                improved_answer, question, intent_analysis
            )

            if improved_quality > quality_score:
                answer = improved_answer
                improvement_count += 1
                self.stats["quality_improvement"] += 1

        # 6단계: 문법 및 구조 개선
        grammar_improved_answer = self.data_processor.fix_grammatical_structure(answer)
        if grammar_improved_answer != answer:
            answer = grammar_improved_answer
            improvement_count += 1
            self.stats["grammar_fix_count"] += 1

        # 7단계: 답변 구조 최적화
        structure_improved_answer = self._optimize_answer_structure(
            answer, intent_analysis
        )
        if structure_improved_answer != answer:
            answer = structure_improved_answer
            improvement_count += 1
            self.stats["answer_structure_improvements"] += 1

        # 8단계: 길이 최적화
        answer = self._optimize_answer_length(answer)
        if answer != original_answer:
            self.stats["answer_length_optimization"] += 1

        # 9단계: 최종 정규화
        answer = self.data_processor.normalize_korean_answer(
            answer, question_type, max_choice
        )

        # 성공 통계 업데이트
        if improvement_count == 0:
            self.stats["fallback_avoidance"] += 1

        self.stats["model_success"] += 1
        self.stats["korean_compliance"] += 1

        # 품질 점수 기록
        final_quality = self._calculate_enhanced_quality_score(
            answer, question, intent_analysis
        )
        self.stats["quality_scores"].append(final_quality)

        # 의도별 품질 통계
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            if primary_intent not in self.stats["answer_quality_by_intent"]:
                self.stats["answer_quality_by_intent"][primary_intent] = []
            self.stats["answer_quality_by_intent"][primary_intent].append(final_quality)

        return answer

    def _optimize_answer_structure(
        self, answer: str, intent_analysis: Dict = None
    ) -> str:
        """답변 구조 최적화"""
        if not answer or len(answer) < 20:
            return answer

        # 의도별 구조 개선
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")

            # 기관명 답변 구조 개선
            if answer_type == "기관명":
                if not any(
                    word in answer for word in ["위원회", "감독원", "은행", "기관"]
                ):
                    # 기관명이 없으면 문장 구조 개선
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

                # 첫 번째 문장이 아니고 접속어가 없으면 추가
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

    def _get_enhanced_improved_answer_with_llm(
        self,
        question: str,
        domain: str,
        intent_analysis: Dict = None,
        kb_analysis: Dict = None,
        improvement_type: str = "general",
    ) -> str:
        """강화된 개선 답변 생성 - 템플릿 예시 활용"""

        # 개선 힌트 정보 수집
        improvement_hints = {"improvement_type": improvement_type, "domain": domain}

        # 기관 관련 질문 힌트
        if kb_analysis and kb_analysis.get("institution_info", {}).get(
            "is_institution_question", False
        ):
            institution_type = kb_analysis["institution_info"].get("institution_type")
            if institution_type:
                improvement_hints["institution_hints"] = (
                    self.knowledge_base.get_institution_hints(institution_type)
                )

        # 의도별 개선 힌트 및 템플릿 예시
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")

            if "기관" in primary_intent:
                intent_key = "기관_묻기"
            elif "특징" in primary_intent:
                intent_key = "특징_묻기"
            elif "지표" in primary_intent:
                intent_key = "지표_묻기"
            elif "방안" in primary_intent:
                intent_key = "방안_묻기"
            elif "절차" in primary_intent:
                intent_key = "절차_묻기"
            elif "조치" in primary_intent:
                intent_key = "조치_묻기"
            else:
                intent_key = "일반"

            # 템플릿 예시 추가
            template_examples = self.knowledge_base.get_template_examples(domain, intent_key)
            if template_examples:
                improvement_hints["template_examples"] = template_examples
                improvement_hints["intent_specific"] = True
                self.stats["template_examples_used"] += 1

        # 강화된 LLM을 통한 개선된 답변 생성
        answer = self.model_handler.generate_improved_answer(
            question, "subjective", 5, intent_analysis, improvement_hints
        )

        # 개선된 답변에 대한 추가 후처리
        answer = self._post_process_improved_answer(
            answer, improvement_type, intent_analysis
        )

        self.stats["llm_usage_rate"] += 1
        self.stats["llm_generation_count"] += 1
        return answer

    def _post_process_improved_answer(
        self, answer: str, improvement_type: str, intent_analysis: Dict = None
    ) -> str:
        """개선된 답변 후처리"""
        if not answer:
            return answer

        # 개선 유형별 후처리
        if improvement_type == "korean_ratio_low":
            # 한국어 비율 개선을 위한 추가 처리
            answer = self.data_processor.enhance_korean_text_quality(answer)

        elif improvement_type == "intent_mismatch":
            # 의도 불일치 개선을 위한 처리
            if intent_analysis:
                answer_type = intent_analysis.get("answer_type_required", "설명형")
                if answer_type == "기관명" and not any(
                    word in answer for word in ["위원회", "감독원", "기관"]
                ):
                    answer = "관련 기관에서 " + answer

        elif improvement_type == "quality_low":
            # 품질 개선을 위한 처리
            if len(answer) < 50:
                answer += " 관련 법령과 규정을 준수하여 체계적으로 관리해야 합니다."

        # 공통 후처리
        answer = self.data_processor.fix_grammatical_structure(answer)

        return answer

    def _calculate_enhanced_quality_score(
        self, answer: str, question: str, intent_analysis: Dict = None
    ) -> float:
        """강화된 품질 점수 계산"""
        if not answer:
            return 0.0

        score = 0.0

        # 한국어 비율 (25%)
        korean_ratio = self.data_processor.calculate_korean_ratio(answer)
        score += korean_ratio * 0.25

        # 텍스트 복구 품질 (10%)
        # 깨진 문자가 없고 자연스러운 한국어인지 확인
        has_broken_chars = any(
            char in answer for char in ["ト", "リ", "ス", "ン", "윋", "젂", "엯"]
        )
        if not has_broken_chars:
            score += 0.1

        # 길이 적절성 (15%)
        length = len(answer)
        if 50 <= length <= 400:
            score += 0.15
        elif 30 <= length < 50 or 400 < length <= 500:
            score += 0.1
        elif 20 <= length < 30:
            score += 0.05

        # 문장 구조 (15%)
        if answer.endswith((".", "다", "요", "함")):
            score += 0.1

        sentences = answer.split(".")
        if len(sentences) >= 2:
            score += 0.05

        # 전문성 (10%)
        domain_keywords = self.model_handler._get_domain_keywords(question)
        found_keywords = sum(1 for keyword in domain_keywords if keyword in answer)
        if found_keywords > 0:
            score += min(found_keywords / len(domain_keywords), 1.0) * 0.1

        # 의도 일치성 (25%)
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            intent_match = self.data_processor.validate_answer_intent_match(
                answer, question, intent_analysis
            )
            if intent_match:
                score += 0.25
            else:
                score += 0.1
        else:
            score += 0.15

        return min(score, 1.0)

    def _optimize_answer_length(self, answer: str) -> str:
        """답변 길이 최적화"""
        if not answer:
            return answer

        # 너무 긴 답변 축약
        if len(answer) > 400:
            sentences = answer.split(". ")
            if len(sentences) > 3:
                answer = ". ".join(sentences[:3])
                if not answer.endswith("."):
                    answer += "."

        # 너무 짧은 답변 보강
        elif len(answer) < 50:
            if not answer.endswith("."):
                answer += "."
            if "법령" not in answer and "규정" not in answer:
                answer += " 관련 법령과 규정을 준수하여 체계적으로 관리해야 합니다."

        return answer

    def _update_mc_stats(
        self,
        question_type: str,
        domain: str,
        difficulty: str,
        processing_time: float,
        answer: str,
        max_choice: int,
    ):
        """객관식 통계 업데이트"""
        self._update_stats(question_type, domain, difficulty, processing_time)

        # 컨텍스트 정확도 추적
        if (
            answer
            and answer.isdigit()
            and max_choice > 0
            and 1 <= int(answer) <= max_choice
        ):
            self.stats["mc_context_accuracy"] += 1

    def _update_subj_stats(
        self,
        question_type: str,
        domain: str,
        difficulty: str,
        processing_time: float,
        intent_analysis: Dict = None,
        answer: str = "",
    ):
        """주관식 통계 업데이트"""
        self._update_stats(question_type, domain, difficulty, processing_time)

        # 도메인별 의도 일치율
        if intent_analysis and question_type == "subjective":
            if domain not in self.stats["domain_intent_match"]:
                self.stats["domain_intent_match"][domain] = {"total": 0, "matched": 0}

            self.stats["domain_intent_match"][domain]["total"] += 1

            # 답변이 의도와 일치하는지 확인
            intent_match = self.data_processor.validate_answer_intent_match(
                answer, "", intent_analysis
            )
            if intent_match:
                self.stats["domain_intent_match"][domain]["matched"] += 1

        # 템플릿 효과성
        if question_type == "subjective" and intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            template_key = f"{domain}_{primary_intent}"

            if template_key not in self.stats["template_effectiveness"]:
                self.stats["template_effectiveness"][template_key] = {
                    "usage": 0,
                    "avg_quality": 0.0,
                    "korean_ratio": 0.0,
                }

            effectiveness = self.stats["template_effectiveness"][template_key]
            effectiveness["usage"] += 1

            if answer:
                quality = self._calculate_enhanced_quality_score(
                    answer, "", intent_analysis
                )
                korean_ratio = self.data_processor.calculate_korean_ratio(answer)

                effectiveness["avg_quality"] = (
                    effectiveness["avg_quality"] * (effectiveness["usage"] - 1)
                    + quality
                ) / effectiveness["usage"]
                effectiveness["korean_ratio"] = (
                    effectiveness["korean_ratio"] * (effectiveness["usage"] - 1)
                    + korean_ratio
                ) / effectiveness["usage"]

    def _get_enhanced_safe_mc_answer_with_llm(
        self, question: str, max_choice: int, domain: str = "일반"
    ) -> str:
        """강화된 안전한 객관식 답변 생성"""
        if max_choice <= 0:
            max_choice = 5

        # 강화된 LLM을 통한 안전한 답변 생성
        fallback_answer = self.model_handler.generate_fallback_mc_answer(
            question, max_choice, domain
        )

        # LLM 결과가 유효하지 않은 경우에만 최후 수단 사용
        if not (
            fallback_answer
            and fallback_answer.isdigit()
            and 1 <= int(fallback_answer) <= max_choice
        ):
            import random

            fallback_answer = str(random.randint(1, max_choice))

        self.stats["llm_usage_rate"] += 1
        self.stats["llm_generation_count"] += 1
        return fallback_answer

    def _get_enhanced_safe_fallback_with_llm(
        self, question: str, question_type: str, max_choice: int
    ) -> str:
        """강화된 안전한 폴백 답변"""
        if max_choice <= 0:
            max_choice = 5

        # 간단한 객관식/주관식 구분
        if question_type == "multiple_choice" or (
            any(str(i) in question for i in range(1, 6)) and len(question) < 300
        ):
            return self._get_enhanced_safe_mc_answer_with_llm(question, max_choice)
        else:
            # 강화된 LLM을 통한 주관식 폴백 답변
            fallback_answer = self.model_handler.generate_fallback_subjective_answer(
                question
            )
            if not fallback_answer or len(fallback_answer) < 20:
                fallback_answer = "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."

            # 추가 품질 개선
            fallback_answer = self.data_processor.clean_korean_text(fallback_answer)

            self.stats["llm_usage_rate"] += 1
            self.stats["llm_generation_count"] += 1
            return fallback_answer

    def _update_stats(
        self, question_type: str, domain: str, difficulty: str, processing_time: float
    ):
        """통계 업데이트"""
        self.stats["total"] += 1
        self.stats["processing_times"].append(processing_time)

        if question_type == "multiple_choice":
            self.stats["mc_count"] += 1
        else:
            self.stats["subj_count"] += 1

        # 도메인 통계
        self.stats["domain_stats"][domain] = (
            self.stats["domain_stats"].get(domain, 0) + 1
        )

        # 난이도 통계
        self.stats["difficulty_stats"][difficulty] += 1

    def print_progress_bar(
        self,
        current: int,
        total: int,
        start_time: float,
        bar_length: int = PROGRESS_CONFIG["bar_length"],
    ):
        """진행률 게이지바 출력"""
        progress = current / total
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        percent = progress * 100
        print(
            f"\r문항 처리: ({current}/{total}) 진행도: {percent:.0f}% [{bar}]",
            end="",
            flush=True,
        )

    def _calculate_model_reliability(self) -> float:
        """모델 신뢰도 계산"""
        total = max(self.stats["total"], 1)

        # 객관식 성공률 (35%)
        mc_total = max(self.stats["mc_count"], 1)
        mc_success_rate = (self.stats["mc_context_accuracy"] / mc_total) * 0.35

        # 주관식 품질 (25%)
        subj_total = max(self.stats["subj_count"], 1)
        subj_quality = 0.0
        if self.stats["quality_scores"]:
            avg_quality = sum(self.stats["quality_scores"]) / len(
                self.stats["quality_scores"]
            )
            subj_quality = avg_quality * 0.25

        # 한국어 준수율 (15%)
        korean_rate = (self.stats["korean_compliance"] / total) * 0.15

        # 텍스트 복구 성공률 (10%)
        recovery_rate = 0.0
        if self.stats["text_recovery_count"] > 0:
            recovery_rate = min(self.stats["text_recovery_count"] / total, 1.0) * 0.1

        # 범위 정확도 (5%)
        range_accuracy = max(0, (1 - self.stats["choice_range_errors"] / total)) * 0.05

        # 검증 통과율 (5%)
        validation_rate = max(0, (1 - self.stats["validation_errors"] / total)) * 0.05

        # LLM 사용률 (5%)
        llm_usage_rate = (self.stats["llm_usage_rate"] / total) * 0.05

        # 전체 신뢰도 (0-100%)
        reliability = (
            mc_success_rate
            + subj_quality
            + korean_rate
            + recovery_rate
            + range_accuracy
            + validation_rate
            + llm_usage_rate
        ) * 100

        return min(reliability, 100.0)

    def _simple_save_csv(self, df: pd.DataFrame, filepath: str) -> bool:
        """간단한 CSV 저장"""
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

        print(f"\n데이터 로드 완료: {len(test_df)}개 문항")

        answers = []
        total_questions = len(test_df)
        inference_start_time = time.time()

        for idx, row in test_df.iterrows():
            question = row["Question"]
            question_id = row["ID"]

            # 추론 수행
            answer = self.process_single_question(question, question_id)
            answers.append(answer)

            # 진행도 표시
            if (idx + 1) % PROGRESS_CONFIG["update_frequency"] == 0:
                self.print_progress_bar(idx + 1, total_questions, inference_start_time)

            # 메모리 관리
            if (idx + 1) % MEMORY_CONFIG["gc_frequency"] == 0:
                gc.collect()

        print()

        # 결과 저장
        submission_df["Answer"] = answers
        save_success = self._simple_save_csv(submission_df, output_file)

        if not save_success:
            print(f"지정된 파일로 저장에 실패했습니다: {output_file}")
            print("파일이 다른 프로그램에서 열려있거나 권한 문제일 수 있습니다.")

        return self._get_results_summary()

    def _print_enhanced_stats(self):
        """상세 통계 출력"""
        pass

    def _get_results_summary(self) -> Dict:
        """결과 요약"""
        total = max(self.stats["total"], 1)
        mc_stats = self.model_handler.get_answer_stats()
        learning_stats = self.model_handler.get_learning_stats()
        processing_stats = self.data_processor.get_processing_stats()
        kb_stats = self.knowledge_base.get_analysis_statistics()

        # 의도별 품질 평균 계산
        intent_quality_avg = {}
        for intent, scores in self.stats["answer_quality_by_intent"].items():
            if scores:
                intent_quality_avg[intent] = sum(scores) / len(scores)

        # 도메인별 의도 일치율 계산
        domain_intent_rates = {}
        for domain, stats in self.stats["domain_intent_match"].items():
            if stats["total"] > 0:
                domain_intent_rates[domain] = (stats["matched"] / stats["total"]) * 100

        # 도메인별 객관식 정확도 계산
        mc_domain_rates = {}
        for domain, stats in self.stats["mc_domain_accuracy"].items():
            if stats["total"] > 0:
                mc_domain_rates[domain] = (stats["success"] / stats["total"]) * 100

        return {
            "success": True,
            "total_questions": self.stats["total"],
            "mc_count": self.stats["mc_count"],
            "subj_count": self.stats["subj_count"],
            "model_success_rate": (self.stats["model_success"] / total) * 100,
            "korean_compliance_rate": (self.stats["korean_compliance"] / total) * 100,
            "choice_range_error_rate": (self.stats["choice_range_errors"] / total)
            * 100,
            "validation_error_rate": (self.stats["validation_errors"] / total) * 100,
            "intent_match_success_rate": (
                self.stats["intent_match_success"]
                / max(self.stats["intent_analysis_accuracy"], 1)
            )
            * 100,
            "institution_questions_count": self.stats["institution_questions"],
            "template_usage_rate": (self.stats["template_usage"] / total) * 100,
            "llm_usage_rate": (self.stats["llm_usage_rate"] / total) * 100,
            "hint_usage_rate": (self.stats["hint_usage_rate"] / total) * 100,
            "text_recovery_rate": (self.stats["text_recovery_count"] / total) * 100,
            "grammar_fix_rate": (self.stats["grammar_fix_count"] / total) * 100,
            "korean_ratio_improvement_rate": (
                self.stats["korean_ratio_improvements"] / total
            )
            * 100,
            "answer_structure_improvement_rate": (
                self.stats["answer_structure_improvements"] / total
            )
            * 100,
            "template_examples_usage_rate": (self.stats["template_examples_used"] / max(self.stats["subj_count"], 1)) * 100,
            "llm_generation_rate": (self.stats["llm_generation_count"] / total) * 100,
            "template_guided_answer_rate": (self.stats["template_guided_answers"] / max(self.stats["subj_count"], 1)) * 100,
            "avg_processing_time": (
                sum(self.stats["processing_times"])
                / len(self.stats["processing_times"])
                if self.stats["processing_times"]
                else 0
            ),
            "avg_quality_score": (
                sum(self.stats["quality_scores"]) / len(self.stats["quality_scores"])
                if self.stats["quality_scores"]
                else 0
            ),
            "intent_quality_by_type": intent_quality_avg,
            "domain_stats": dict(self.stats["domain_stats"]),
            "difficulty_stats": dict(self.stats["difficulty_stats"]),
            "answer_distribution_by_range": self.stats["mc_answers_by_range"],
            "learning_stats": learning_stats,
            "processing_stats": processing_stats,
            "knowledge_base_stats": kb_stats,
            "mc_context_accuracy_rate": (
                self.stats["mc_context_accuracy"] / max(self.stats["mc_count"], 1)
            )
            * 100,
            "mc_pattern_match_rate": (
                self.stats["mc_pattern_matches"] / max(self.stats["mc_count"], 1)
            )
            * 100,
            "high_confidence_intent_rate": (
                self.stats["high_confidence_intent"]
                / max(self.stats["intent_analysis_accuracy"], 1)
            )
            * 100,
            "intent_specific_answer_rate": (
                self.stats["intent_specific_answers"] / total
            )
            * 100,
            "quality_improvement_count": self.stats["quality_improvement"],
            "fallback_avoidance_rate": (self.stats["fallback_avoidance"] / total) * 100,
            "korean_enhancement_count": self.stats["korean_enhancement"],
            "answer_length_optimization_count": self.stats[
                "answer_length_optimization"
            ],
            "domain_intent_match_rates": domain_intent_rates,
            "mc_domain_accuracy_rates": mc_domain_rates,
            "institution_answer_accuracy": self.stats["institution_answer_accuracy"],
            "template_effectiveness_stats": dict(self.stats["template_effectiveness"]),
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
            print(f"모델 성공률: {results['model_success_rate']:.1f}%")
            print(f"한국어 준수율: {results['korean_compliance_rate']:.1f}%")
            print(f"LLM 사용률: {results['llm_usage_rate']:.1f}%")
            print(f"LLM 생성률: {results['llm_generation_rate']:.1f}%")
            print(f"템플릿 예시 활용률: {results['template_examples_usage_rate']:.1f}%")
            print(f"템플릿 가이드 답변률: {results['template_guided_answer_rate']:.1f}%")
            print(f"텍스트 복구율: {results['text_recovery_rate']:.1f}%")
            print(f"문법 수정률: {results['grammar_fix_rate']:.1f}%")
            if results["choice_range_error_rate"] > 0:
                print(f"선택지 범위 오류율: {results['choice_range_error_rate']:.1f}%")
            if results["intent_match_success_rate"] > 0:
                print(f"의도 일치 성공률: {results['intent_match_success_rate']:.1f}%")
            if results["mc_context_accuracy_rate"] > 0:
                print(
                    f"객관식 컨텍스트 정확도: {results['mc_context_accuracy_rate']:.1f}%"
                )

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