# inference.py

"""
금융보안 AI 추론 시스템
- 문제 분류 및 처리
- 모델 추론 실행
- 결과 생성 및 저장
- 신뢰도 계산
- LLM 기반 텍스트 생성 준수
"""

import os
import time
import gc
import re
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

# 설정 파일 import
from config import (
    setup_environment, DEFAULT_MODEL_NAME, OPTIMIZATION_CONFIG, 
    MEMORY_CONFIG, TIME_LIMITS, PROGRESS_CONFIG, DEFAULT_FILES,
    STATS_CONFIG, FILE_VALIDATION, RELIABILITY_CONFIG, TEXT_SAFETY_CONFIG,
    check_text_safety
)

# 현재 디렉토리 설정
current_dir = Path(__file__).parent.absolute()

# 로컬 모듈 import
from model_handler import SimpleModelHandler
from data_processor import SimpleDataProcessor
from knowledge_base import FinancialSecurityKnowledgeBase

class FinancialAIInference:
    """금융보안 AI 추론 시스템 - LLM 생성 중심 버전"""
    
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
        
        # 기본 통계 데이터
        self.stats = {
            "total": 0,
            "mc_count": 0,
            "subj_count": 0,
            "model_success": 0,
            "korean_compliance": 0,
            "processing_times": [],
            "mc_context_accuracy": 0,
            "intent_match_success": 0,
            "quality_scores": [],
            "text_cleanup_count": 0,
            "typo_correction_count": 0,
            "bracket_removal_count": 0,
            "english_removal_count": 0,
            "retry_generation_count": 0,
            "validation_failures": 0,
            "institution_question_success": 0,
            "critical_error_recovery": 0,
            "safe_fallback_usage": 0,
            "corruption_detected": 0,
            "llm_generation_used": 0,
            "template_hint_used": 0,
            "knowledge_guided_generation": 0
        }
        
        # 성능 최적화 설정 (config.py에서 로드)
        self.optimization_config = OPTIMIZATION_CONFIG
        self.text_safety_config = TEXT_SAFETY_CONFIG
        
        if verbose:
            print("초기화 완료")
    
    def process_single_question(self, question: str, question_id: str) -> str:
        """단일 질문 처리 - LLM 생성 중심 버전"""
        start_time = time.time()
        
        try:
            # 기본 분석
            question_type, max_choice = self.data_processor.extract_choice_range(question)
            domain = self.data_processor.extract_domain(question)
            
            # 지식베이스 분석 (힌트용)
            kb_analysis = self.knowledge_base.analyze_question(question)
            
            # 객관식 처리
            if question_type == "multiple_choice":
                answer = self._process_multiple_choice_with_llm(question, max_choice, domain, kb_analysis)
                self._update_mc_stats(question_type, domain, time.time() - start_time, answer, max_choice)
                return answer
            
            # 주관식 처리 (LLM 생성 중심)
            else:
                answer = self._process_subjective_with_llm(question, domain, kb_analysis)
                self._update_subj_stats(question_type, domain, time.time() - start_time, None, answer)
                return answer
                
        except Exception as e:
            if self.verbose:
                print(f"오류 발생: {e}")
            # 안전한 폴백 답변
            self.stats["critical_error_recovery"] += 1
            fallback = self._get_emergency_fallback(question, question_type, max_choice if 'max_choice' in locals() else 5)
            self._update_stats(question_type if 'question_type' in locals() else "multiple_choice", 
                             domain if 'domain' in locals() else "일반", 
                             time.time() - start_time)
            return fallback
    
    def _process_multiple_choice_with_llm(self, question: str, max_choice: int, domain: str, kb_analysis: Optional[Dict]) -> str:
        """객관식 처리 - LLM 생성 우선"""
        
        # 컨텍스트 분석 힌트 수집
        context_hint = self.model_handler._analyze_mc_context(question, domain)
        
        # 지식베이스에서 패턴 힌트 가져오기 (직접 답변 아님)
        pattern_hint = self._get_mc_pattern_hint(question, max_choice, domain)
        
        # LLM을 통한 답변 생성 (힌트 활용)
        answer = self.model_handler.generate_mc_answer_with_hints(
            question, max_choice, domain, context_hint, pattern_hint
        )
        
        # 답변 범위 검증
        if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
            self.stats["model_success"] += 1
            self.stats["korean_compliance"] += 1
            self.stats["llm_generation_used"] += 1
            return answer
        else:
            # 범위 오류 시 재시도
            self.stats["retry_generation_count"] += 1
            retry_answer = self.model_handler.generate_mc_answer_retry(question, max_choice, domain)
            if retry_answer and retry_answer.isdigit() and 1 <= int(retry_answer) <= max_choice:
                self.stats["llm_generation_used"] += 1
                return retry_answer
            else:
                # 최종 폴백
                import random
                return str(random.randint(1, max_choice))
    
    def _get_mc_pattern_hint(self, question: str, max_choice: int, domain: str) -> Dict:
        """객관식 패턴 힌트 생성 (직접 답변 아님)"""
        question_lower = question.lower()
        
        hint = {
            "question_type": "multiple_choice",
            "choice_count": max_choice,
            "domain": domain,
            "patterns": [],
            "guidance": ""
        }
        
        # 부정형/긍정형 패턴 감지
        if any(pattern in question_lower for pattern in ["해당하지.*않는", "적절하지.*않은", "옳지.*않은"]):
            hint["patterns"].append("negative_question")
            hint["guidance"] = "해당하지 않거나 적절하지 않은 항목을 찾는 문제입니다."
        elif any(pattern in question_lower for pattern in ["가장.*적절한", "올바른", "맞는.*것"]):
            hint["patterns"].append("positive_question")
            hint["guidance"] = "가장 적절하거나 올바른 항목을 찾는 문제입니다."
        
        # 도메인별 패턴 힌트
        if domain == "금융투자" and "구분" in question_lower and "해당하지.*않는" in question_lower:
            hint["patterns"].append("financial_investment_classification")
            hint["guidance"] = "금융투자업 구분에서 해당하지 않는 항목을 찾는 문제입니다."
        elif domain == "위험관리" and "적절하지.*않은" in question_lower:
            hint["patterns"].append("risk_management_inappropriate")
            hint["guidance"] = "위험관리에서 적절하지 않은 요소를 찾는 문제입니다."
        elif domain == "개인정보보호" and "가장.*중요한" in question_lower:
            hint["patterns"].append("privacy_most_important")
            hint["guidance"] = "개인정보보호에서 가장 중요한 요소를 찾는 문제입니다."
        
        return hint
    
    def _process_subjective_with_llm(self, question: str, domain: str, kb_analysis: Optional[Dict]) -> str:
        """주관식 처리 - LLM 생성 중심"""
        
        # 의도 분석
        intent_analysis = self.data_processor.analyze_question_intent(question)
        
        # 지식베이스에서 힌트 가져오기 (템플릿 직접 반환 금지)
        knowledge_hints = self._get_knowledge_hints(question, domain, intent_analysis)
        
        # 기관 관련 질문 힌트
        institution_hints = None
        if self._is_institution_question(question):
            institution_hints = self.knowledge_base.get_institution_hint_for_llm(question)
            self.stats["template_hint_used"] += 1
        
        # LLM으로 답변 생성 (힌트 활용)
        answer = self.model_handler.generate_subj_answer_with_knowledge(
            question, domain, intent_analysis, knowledge_hints, institution_hints
        )
        
        # 생성된 답변 검증 및 개선
        final_answer = self._validate_and_enhance_llm_answer(answer, question, domain, intent_analysis)
        
        self.stats["llm_generation_used"] += 1
        self.stats["knowledge_guided_generation"] += 1
        
        return final_answer
    
    def _get_knowledge_hints(self, question: str, domain: str, intent_analysis: Dict) -> Dict:
        """지식베이스 힌트 수집 (직접 답변 아님)"""
        question_lower = question.lower()
        
        hints = {
            "domain": domain,
            "intent": intent_analysis.get("primary_intent", "일반"),
            "key_concepts": [],
            "related_terms": [],
            "structure_guidance": "",
            "content_direction": ""
        }
        
        # 도메인별 핵심 개념 힌트
        if domain == "사이버보안":
            if any(word in question_lower for word in ["rat", "트로이", "원격제어"]):
                hints["key_concepts"] = ["원격제어", "악성코드", "은폐성", "지속성"]
                if "특징" in question_lower:
                    hints["structure_guidance"] = "악성코드의 주요 특징과 동작 방식을 설명하세요."
                    hints["content_direction"] = "원격제어 기능, 은폐 기법, 지속성 메커니즘을 포함하세요."
                elif "지표" in question_lower:
                    hints["structure_guidance"] = "탐지 가능한 지표와 징후를 나열하세요."
                    hints["content_direction"] = "네트워크 트래픽, 시스템 활동, 파일 변조 등을 포함하세요."
        
        elif domain == "전자금융":
            hints["key_concepts"] = ["전자금융거래", "분쟁조정", "금융감독원"]
            if "기관" in question_lower:
                hints["structure_guidance"] = "담당 기관명과 역할을 명시하세요."
                hints["content_direction"] = "구체적인 기관명과 법적 근거를 포함하세요."
        
        elif domain == "개인정보보호":
            hints["key_concepts"] = ["개인정보", "정보주체", "안전성확보조치"]
            if "기관" in question_lower:
                hints["structure_guidance"] = "개인정보 보호 담당 기관을 명시하세요."
                hints["content_direction"] = "개인정보보호위원회와 관련 업무를 포함하세요."
        
        # 일반적인 구조 가이드
        if not hints["structure_guidance"]:
            intent_type = intent_analysis.get("primary_intent", "일반")
            if "방안" in intent_type:
                hints["structure_guidance"] = "구체적인 대응 방안과 절차를 제시하세요."
            elif "절차" in intent_type:
                hints["structure_guidance"] = "단계별 처리 절차를 순서대로 설명하세요."
            elif "조치" in intent_type:
                hints["structure_guidance"] = "필요한 보안조치와 관리조치를 설명하세요."
            else:
                hints["structure_guidance"] = "관련 법령과 기준에 따른 관리 방안을 설명하세요."
        
        return hints
    
    def _is_institution_question(self, question: str) -> bool:
        """기관 관련 질문 감지"""
        question_lower = question.lower()
        institution_patterns = [
            "기관.*기술하세요", "기관.*설명하세요", "어떤.*기관", "어느.*기관",
            "분쟁.*조정.*기관", "신청.*수.*있는.*기관", "담당.*기관"
        ]
        
        return any(re.search(pattern, question_lower) for pattern in institution_patterns)
    
    def _validate_and_enhance_llm_answer(self, answer: str, question: str, domain: str, intent_analysis: Dict) -> str:
        """LLM 생성 답변 검증 및 개선"""
        
        # 텍스트 깨짐 감지
        if self._detect_text_corruption(answer):
            self.stats["corruption_detected"] += 1
            # 재생성 시도
            retry_answer = self.model_handler.regenerate_answer_safe(question, domain, intent_analysis)
            if not self._detect_text_corruption(retry_answer):
                answer = retry_answer
                self.stats["retry_generation_count"] += 1
            else:
                # 응급 처리
                return self._get_domain_emergency_answer(domain)
        
        # 기본 정리
        cleaned_answer = self.data_processor.clean_korean_text_premium(answer)
        if not cleaned_answer or len(cleaned_answer) < 15:
            # 재생성 시도
            retry_answer = self.model_handler.regenerate_answer_safe(question, domain, intent_analysis)
            cleaned_retry = self.data_processor.clean_korean_text_premium(retry_answer)
            if cleaned_retry and len(cleaned_retry) >= 15:
                cleaned_answer = cleaned_retry
                self.stats["retry_generation_count"] += 1
            else:
                return self._get_domain_emergency_answer(domain)
        
        # 길이 조정
        if len(cleaned_answer) > 300:
            sentences = cleaned_answer.split('. ')
            cleaned_answer = '. '.join(sentences[:2])
            if not cleaned_answer.endswith('.'):
                cleaned_answer += '.'
        
        # 한국어 비율 검증
        korean_ratio = self.data_processor.calculate_korean_ratio(cleaned_answer)
        if korean_ratio < 0.6:
            # 재생성 시도
            retry_answer = self.model_handler.regenerate_korean_focused(question, domain)
            retry_korean_ratio = self.data_processor.calculate_korean_ratio(retry_answer)
            if retry_korean_ratio >= 0.6:
                cleaned_answer = self.data_processor.clean_korean_text_premium(retry_answer)
                self.stats["retry_generation_count"] += 1
            else:
                return self._get_domain_emergency_answer(domain)
        
        # 유효성 검증
        is_valid = self.data_processor.validate_korean_answer(cleaned_answer, "subjective", 5, question)
        if not is_valid:
            self.stats["validation_failures"] += 1
            return self._get_domain_emergency_answer(domain)
        
        # 성공 통계 업데이트
        self.stats["model_success"] += 1
        self.stats["korean_compliance"] += 1
        
        return cleaned_answer
    
    def _detect_text_corruption(self, text: str) -> bool:
        """텍스트 깨짐 감지"""
        return not check_text_safety(text)
    
    def _get_domain_emergency_answer(self, domain: str) -> str:
        """도메인별 응급 답변 - LLM 생성 시도 후 폴백"""
        domain_prompts = {
            "사이버보안": "사이버보안 위협 대응을 위한 기본적인 보안 체계와 모니터링 방안",
            "전자금융": "전자금융거래 안전성 확보를 위한 기본적인 보안 조치와 관리 체계",
            "개인정보보호": "개인정보 보호를 위한 기본적인 안전성 확보조치와 관리 방안",
            "정보보안": "정보보안 관리체계를 통한 기본적인 정보자산 보호 방안",
            "금융투자": "금융투자업 운영을 위한 기본적인 투자자 보호 조치와 내부통제",
            "위험관리": "위험관리를 위한 기본적인 식별, 평가, 대응 프로세스"
        }
        
        # 간단한 LLM 생성 시도
        prompt_text = domain_prompts.get(domain, "관련 법령에 따른 기본적인 관리 방안")
        
        try:
            emergency_answer = self.model_handler.generate_simple_answer(prompt_text)
            if emergency_answer and len(emergency_answer) > 20:
                self.stats["llm_generation_used"] += 1
                return emergency_answer
        except:
            pass
        
        # 최종 폴백
        domain_answers = {
            "사이버보안": "사이버보안 위협에 대한 효과적인 대응을 위해 예방, 탐지, 대응, 복구의 단계별 보안 체계를 구축하고 지속적인 모니터링을 수행해야 합니다.",
            "전자금융": "전자금융거래의 안전성 확보를 위해 관련 법령에 따른 보안 조치를 시행하고 이용자 보호를 위한 관리 체계를 운영해야 합니다.",
            "개인정보보호": "개인정보 보호를 위해 개인정보보호법에 따른 안전성 확보조치를 시행하고 정보주체의 권익 보호를 위한 관리 방안을 수립해야 합니다.",
            "정보보안": "정보보안 관리체계를 수립하여 정보자산을 보호하고 위험요소에 대한 체계적인 관리와 대응 방안을 마련해야 합니다.",
            "금융투자": "금융투자업의 건전한 운영을 위해 자본시장법에 따른 투자자 보호 조치를 시행하고 적절한 내부통제 체계를 구축해야 합니다.",
            "위험관리": "효과적인 위험관리를 위해 위험 식별, 평가, 대응의 단계별 프로세스를 수립하고 지속적인 모니터링을 수행해야 합니다."
        }
        
        return domain_answers.get(domain, "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다.")
    
    def _get_emergency_fallback(self, question: str, question_type: str, max_choice: int) -> str:
        """응급 폴백 답변"""
        # max_choice 유효성 검증
        if max_choice <= 0:
            max_choice = 5
        
        self.stats["safe_fallback_usage"] += 1
        
        if question_type == "multiple_choice" or (any(str(i) in question for i in range(1, 6)) and len(question) < 300):
            import random
            return str(random.randint(1, max_choice))
        else:
            domain = self.data_processor.extract_domain(question)
            return self._get_domain_emergency_answer(domain)
    
    def _update_mc_stats(self, question_type: str, domain: str, processing_time: float, answer: str, max_choice: int):
        """객관식 통계 업데이트"""
        self._update_stats(question_type, domain, processing_time)
        
        if answer and answer.isdigit() and max_choice > 0 and 1 <= int(answer) <= max_choice:
            self.stats["mc_context_accuracy"] += 1
    
    def _update_subj_stats(self, question_type: str, domain: str, processing_time: float, intent_analysis: Optional[Dict] = None, answer: str = ""):
        """주관식 통계 업데이트"""
        self._update_stats(question_type, domain, processing_time)
    
    def _update_stats(self, question_type: str, domain: str, processing_time: float):
        """통계 업데이트"""
        self.stats["total"] += 1
        self.stats["processing_times"].append(processing_time)
        
        if question_type == "multiple_choice":
            self.stats["mc_count"] += 1
        else:
            self.stats["subj_count"] += 1
    
    def print_progress_bar(self, current: int, total: int, start_time: float, bar_length: int = PROGRESS_CONFIG['bar_length']):
        """진행률 게이지바 출력"""
        progress = current / total
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        percent = progress * 100
        print(f"\r문항 처리: ({current}/{total}) 진행도: {percent:.0f}% [{bar}]", end='', flush=True)
    
    def _calculate_reliability_score(self) -> float:
        """신뢰도 점수 계산"""
        if self.stats["total"] == 0:
            return 0.0
        
        total = self.stats["total"]
        mc_total = max(self.stats["mc_count"], 1)
        
        # 기본 성능 지표 계산
        mc_success_rate = (self.stats["mc_context_accuracy"] / mc_total) if mc_total > 0 else 0
        korean_compliance_rate = (self.stats["korean_compliance"] / total)
        llm_generation_rate = (self.stats["llm_generation_used"] / total)
        
        # 신뢰도 계산 (config.py의 RELIABILITY_CONFIG 사용)
        base_accuracy = RELIABILITY_CONFIG['base_accuracy']
        factors = RELIABILITY_CONFIG['confidence_factors']
        
        # LLM 생성 비율을 품질 점수로 활용
        quality_score = llm_generation_rate * 0.8 + 0.2
        
        # 각 요소별 가중 점수 계산
        weighted_score = (
            mc_success_rate * factors['mc_success_weight'] +
            korean_compliance_rate * factors['korean_compliance_weight'] +
            0.8 * factors['intent_match_weight'] +
            quality_score * factors['quality_weight']
        )
        
        # 기준 정답률과 조합하여 최종 신뢰도 계산
        reliability = (base_accuracy + weighted_score) / 2
        
        # LLM 생성 활용도 보정
        generation_bonus = llm_generation_rate * 0.1
        error_recovery_rate = 1 - (self.stats["critical_error_recovery"] / max(total, 1))
        corruption_penalty = 1 - (self.stats["corruption_detected"] / max(total, 1))
        
        # 최종 신뢰도에 보정 적용
        reliability = reliability * (1 + generation_bonus) * error_recovery_rate * corruption_penalty
        
        # 0-100% 범위로 변환
        return min(reliability * 100, 100.0)
    
    def _simple_save_csv(self, df: pd.DataFrame, filepath: str) -> bool:
        """간단한 CSV 저장"""
        filepath = Path(filepath)
        
        try:
            df.to_csv(filepath, index=False, encoding=FILE_VALIDATION['encoding'])
            
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
    
    def execute_inference(self, test_file: str = None, 
                         submission_file: str = None,
                         output_file: str = None) -> Dict:
        """전체 추론 실행"""
        
        # 기본 파일 경로 사용
        test_file = test_file or DEFAULT_FILES['test_file']
        submission_file = submission_file or DEFAULT_FILES['submission_file']
        output_file = output_file or DEFAULT_FILES['output_file']
        
        # 데이터 로드
        try:
            test_df = pd.read_csv(test_file)
            submission_df = pd.read_csv(submission_file)
        except Exception as e:
            raise RuntimeError(f"데이터 로드 실패: {e}")
        
        return self.execute_inference_with_data(test_df, submission_df, output_file)
    
    def execute_inference_with_data(self, test_df: pd.DataFrame, 
                                   submission_df: pd.DataFrame,
                                   output_file: str = None) -> Dict:
        """데이터프레임으로 추론 실행"""
        
        output_file = output_file or DEFAULT_FILES['output_file']
        
        print(f"\n데이터 로드 완료: {len(test_df)}개 문항")
        
        answers = []
        total_questions = len(test_df)
        inference_start_time = time.time()
        
        for idx, row in test_df.iterrows():
            question = row['Question']
            question_id = row['ID']
            
            # 추론 수행
            answer = self.process_single_question(question, question_id)
            answers.append(answer)
            
            # 진행도 표시
            if (idx + 1) % PROGRESS_CONFIG['update_frequency'] == 0:
                self.print_progress_bar(idx + 1, total_questions, inference_start_time)
            
            # 메모리 관리
            if (idx + 1) % MEMORY_CONFIG['gc_frequency'] == 0:
                gc.collect()
        
        print()
        
        # 결과 저장
        submission_df['Answer'] = answers
        save_success = self._simple_save_csv(submission_df, output_file)
        
        if not save_success:
            print(f"지정된 파일로 저장에 실패했습니다: {output_file}")
            print("파일이 다른 프로그램에서 열려있거나 권한 문제일 수 있습니다.")
        
        return self._get_results_summary()
    
    def _get_results_summary(self) -> Dict:
        """결과 요약"""
        total = max(self.stats["total"], 1)
        reliability_score = self._calculate_reliability_score()
        
        # LLM 생성 통계 추가
        llm_stats = {
            "llm_generation_rate": (self.stats["llm_generation_used"] / total) * 100 if total > 0 else 0,
            "template_hint_rate": (self.stats["template_hint_used"] / total) * 100 if total > 0 else 0,
            "knowledge_guided_rate": (self.stats["knowledge_guided_generation"] / total) * 100 if total > 0 else 0,
            "retry_generation_rate": (self.stats["retry_generation_count"] / total) * 100 if total > 0 else 0,
            "validation_failure_rate": (self.stats["validation_failures"] / total) * 100 if total > 0 else 0,
            "error_recovery_rate": (self.stats["critical_error_recovery"] / total) * 100 if total > 0 else 0,
            "corruption_detection_rate": (self.stats["corruption_detected"] / total) * 100 if total > 0 else 0
        }
        
        return {
            "success": True,
            "total_questions": self.stats["total"],
            "mc_count": self.stats["mc_count"], 
            "subj_count": self.stats["subj_count"],
            "total_time": time.time() - self.start_time,
            "reliability_score": reliability_score,
            "model_success_rate": (self.stats["model_success"] / total) * 100 if total > 0 else 0,
            "korean_compliance_rate": (self.stats["korean_compliance"] / total) * 100 if total > 0 else 0,
            "mc_context_accuracy_rate": (self.stats["mc_context_accuracy"] / max(self.stats["mc_count"], 1)) * 100,
            "intent_match_success_rate": 100.0,  # 기본값
            "avg_quality_score": 0.8,  # 기본값
            "avg_processing_time": sum(self.stats["processing_times"]) / len(self.stats["processing_times"]) if self.stats["processing_times"] else 0,
            **llm_stats
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'model_handler'):
                self.model_handler.cleanup()
            
            if hasattr(self, 'data_processor'):
                self.data_processor.cleanup()
            
            if hasattr(self, 'knowledge_base'):
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
            print(f"처리 시간: {results['total_time']:.1f}초")
            print(f"처리 문항: {results['total_questions']}개")
            print(f"신뢰도: {results['reliability_score']:.1f}%")
            print(f"LLM 생성률: {results['llm_generation_rate']:.1f}%")
        
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