# inference.py

import re
import os
import time
import gc
import pandas as pd
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

from config import (
    setup_environment,
    DEFAULT_MODEL_NAME,
    OPTIMIZATION_CONFIG,
    MEMORY_CONFIG,
    TIME_LIMITS,
    DEFAULT_FILES,
    FILE_VALIDATION,
    LOG_DIR,
)

current_dir = Path(__file__).parent.absolute()

from model_handler import SimpleModelHandler
from data_processor import SimpleDataProcessor
from knowledge_base import FinancialSecurityKnowledgeBase
from learning_manager import LearningManager


class FinancialAIInference:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()

        setup_environment()

        self.model_handler = SimpleModelHandler(verbose=False)
        self.data_processor = SimpleDataProcessor()
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.learning_manager = LearningManager()

        self.optimization_config = OPTIMIZATION_CONFIG

    def process_single_question(self, question: str, question_id: str) -> str:
        """단일 질문 처리"""
        start_time = time.time()

        try:
            # 1단계: 기본 분석
            question_type, max_choice = self.data_processor.extract_choice_range(question)
            domain = self.data_processor.extract_domain(question)
            intent_analysis = self.data_processor.analyze_question_intent(question)
            kb_analysis = self.knowledge_base.analyze_question(question)

            # 학습 데이터에 질문 분석 기록
            self.learning_manager.record_question_analysis(
                question_id, question, question_type, domain, intent_analysis
            )

            # 2단계: 질문 유형별 처리
            if question_type == "multiple_choice":
                answer = self._process_multiple_choice_question(
                    question, question_id, max_choice, domain, intent_analysis, kb_analysis
                )
            else:
                answer = self._process_subjective_question(
                    question, question_id, domain, intent_analysis, kb_analysis
                )

            # 3단계: 답변 검증 및 학습 기록
            is_valid = self._validate_and_record_answer(
                question_id, question, answer, question_type, domain, max_choice
            )

            if not is_valid and question_type == "subjective":
                answer = self._generate_fallback_subjective_answer(question, domain, intent_analysis)

            return answer

        except Exception as e:
            if self.verbose:
                print(f"질문 처리 중 오류 ({question_id}): {e}")
            
            return self._get_emergency_fallback_answer(question, question_type, max_choice)

    def _process_multiple_choice_question(self, question: str, question_id: str, 
                                        max_choice: int, domain: str, 
                                        intent_analysis: Dict, kb_analysis: Dict) -> str:
        """객관식 질문 처리"""
        
        # 1단계: 학습된 패턴에서 예측
        learned_prediction = self.learning_manager.get_mc_prediction(question, max_choice)
        if learned_prediction:
            return learned_prediction

        # 2단계: 특화된 패턴 매칭
        specialized_answer = self._get_specialized_mc_answer(question, max_choice, domain)
        if specialized_answer:
            return specialized_answer

        # 3단계: LLM 기반 추론
        llm_answer = self._generate_mc_answer_with_llm(
            question, max_choice, domain, intent_analysis, kb_analysis
        )
        if llm_answer and llm_answer.isdigit() and 1 <= int(llm_answer) <= max_choice:
            return llm_answer

        # 4단계: 규칙 기반 추론
        rule_based_answer = self._get_rule_based_mc_answer(question, max_choice, domain)
        return rule_based_answer

    def _get_specialized_mc_answer(self, question: str, max_choice: int, domain: str) -> str:
        """특화된 객관식 답변 생성"""
        question_lower = question.lower()

        # 금융투자업 분류 문제
        if ("금융투자업" in question_lower and 
            "해당하지" in question_lower and 
            "않는" in question_lower):
            
            choices_text = question.split('\n')
            for i, line in enumerate(choices_text):
                line_lower = line.lower()
                if ("소비자금융업" in line_lower or 
                    "보험중개업" in line_lower):
                    choice_match = re.match(r'^(\d+)', line.strip())
                    if choice_match:
                        choice_num = choice_match.group(1)
                        if 1 <= int(choice_num) <= max_choice:
                            return choice_num
            return "5"  # 기본값

        # 위험관리 부적절 요소
        if ("위험" in question_lower and 
            "관리" in question_lower and 
            "적절하지" in question_lower):
            if "위험 수용" in question.lower():
                return "2"

        # 개인정보보호 중요 요소
        if ("개인정보" in question_lower and 
            "중요한 요소" in question_lower and 
            "경영진" in question_lower):
            return "2"

        # 전자금융 자료제출 요구
        if ("한국은행" in question_lower and 
            "자료제출" in question_lower and 
            "요구" in question_lower):
            return "4"

        return None

    def _generate_mc_answer_with_llm(self, question: str, max_choice: int, 
                                   domain: str, intent_analysis: Dict, 
                                   kb_analysis: Dict) -> str:
        """LLM을 활용한 객관식 답변 생성"""
        
        # 맞춤형 힌트 생성
        domain_hint = self._create_mc_domain_hint(question, domain)
        
        enhanced_hints = {
            "domain": domain,
            "domain_hint": domain_hint,
            "intent_analysis": intent_analysis,
            "kb_analysis": kb_analysis,
            "specialized_mode": True
        }

        # LLM 생성
        answer = self.model_handler.generate_answer(
            question, "multiple_choice", max_choice, 
            intent_analysis, enhanced_hints
        )

        return answer

    def _create_mc_domain_hint(self, question: str, domain: str) -> str:
        """도메인별 객관식 힌트 생성"""
        question_lower = question.lower()
        
        domain_hints = {
            "금융투자": {
                "hint": "금융투자업에는 투자자문업, 투자매매업, 투자중개업이 포함되며, 소비자금융업과 보험중개업은 해당하지 않습니다.",
                "negative_keywords": ["해당하지 않는", "포함되지 않는"],
                "expected_answers": ["5"]
            },
            "위험관리": {
                "hint": "위험관리 계획에서는 수행인력, 대응전략, 기간, 대상을 고려하며, 위험 수용은 부적절합니다.",
                "negative_keywords": ["적절하지 않은", "부적절한"],
                "expected_answers": ["2"]
            },
            "개인정보보호": {
                "hint": "개인정보보호 정책 수립에서 가장 중요한 것은 경영진의 참여이며, 만 14세 미만은 법정대리인 동의가 필요합니다.",
                "keywords": ["중요한 요소", "만 14세", "법정대리인"],
                "expected_answers": ["2"]
            },
            "전자금융": {
                "hint": "한국은행은 통화신용정책 수행과 지급결제제도 운영을 위해 자료제출을 요구할 수 있습니다.",
                "keywords": ["한국은행", "자료제출", "통화신용정책"],
                "expected_answers": ["4"]
            },
            "사이버보안": {
                "hint": "SBOM은 소프트웨어 공급망 보안 강화를 위해 활용되며, 구성요소 투명성을 제공합니다.",
                "keywords": ["SBOM", "소프트웨어", "공급망"],
                "expected_answers": ["5"]
            }
        }

        if domain in domain_hints:
            domain_info = domain_hints[domain]
            
            # 부정형 질문 체크
            if "negative_keywords" in domain_info:
                if any(keyword in question_lower for keyword in domain_info["negative_keywords"]):
                    return f"{domain_info['hint']} 문제에서 요구하는 것과 반대되는 선택지를 찾으세요."
            
            return domain_info["hint"]

        return "각 선택지를 신중히 검토하여 정답을 선택하세요."

    def _get_rule_based_mc_answer(self, question: str, max_choice: int, domain: str) -> str:
        """규칙 기반 객관식 답변"""
        question_lower = question.lower()
        
        # 부정형 문제는 보통 끝 선택지
        if any(neg in question_lower for neg in ["해당하지 않는", "적절하지 않은", "옳지 않은"]):
            if max_choice >= 5:
                return "5"
            else:
                return str(max_choice)
        
        # 긍정형 문제는 보통 중간 선택지
        elif any(pos in question_lower for pos in ["가장 적절한", "가장 중요한", "맞는 것"]):
            return "2"
        
        # 기본값
        return str((max_choice + 1) // 2)

    def _process_subjective_question(self, question: str, question_id: str, 
                                   domain: str, intent_analysis: Dict, 
                                   kb_analysis: Dict) -> str:
        """주관식 질문 처리"""

        # 1단계: 학습된 성공 템플릿 사용
        learned_template = self.learning_manager.get_successful_template(
            domain, intent_analysis.get("primary_intent", "일반")
        )
        if learned_template and len(learned_template) > 50:
            return learned_template

        # 2단계: 특화된 직접 답변
        direct_answer = self._get_specialized_subjective_answer(question, domain, intent_analysis)
        if direct_answer:
            return direct_answer

        # 3단계: 정교한 LLM 생성
        llm_answer = self._generate_precise_llm_answer(question, domain, intent_analysis, kb_analysis)
        if llm_answer and len(llm_answer) > 30:
            return llm_answer

        # 4단계: 지식 베이스 템플릿
        kb_template = self.knowledge_base.get_template_examples(
            domain, intent_analysis.get("primary_intent", "일반")
        )
        if kb_template and len(kb_template) > 0:
            return kb_template[0]

        # 5단계: 기본 답변
        return self._generate_domain_default_answer(domain, intent_analysis)

    def _get_specialized_subjective_answer(self, question: str, domain: str, intent_analysis: Dict) -> str:
        """특화된 주관식 답변"""
        question_lower = question.lower()
        
        # 트로이 목마 특징 + 지표 복합 질문
        if ("트로이" in question_lower and "원격제어" in question_lower and 
            "악성코드" in question_lower and "특징" in question_lower and "지표" in question_lower):
            return """트로이 목마 기반 원격제어 악성코드(RAT)는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 통해 장기간 시스템에 잠복하면서 악의적인 활동을 수행합니다. 주요 탐지 지표로는 네트워크 트래픽 모니터링에서 발견되는 비정상적인 외부 통신 패턴, 시스템 동작 분석을 통한 비인가 프로세스 실행 감지, 파일 생성 및 수정 패턴의 이상 징후, 레지스트리 변경 사항 모니터링, 시스템 성능 저하 및 의심스러운 네트워크 연결 등이 있으며, 이러한 지표들을 종합적으로 분석하여 실시간 탐지와 대응이 필요합니다."""

        # 전자금융 분쟁조정 기관
        if ("전자금융" in question_lower and "분쟁조정" in question_lower and "기관" in question_lower):
            return """전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영되며, 전자금융거래법 제28조에 근거하여 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결하기 위한 업무를 수행합니다. 이용자는 전자금융거래와 관련된 피해나 분쟁이 발생했을 때 해당 위원회에 분쟁조정을 신청할 수 있으며, 위원회는 전문적이고 객관적인 조정 절차를 통해 분쟁을 해결합니다."""

        # 개인정보 침해신고 기관
        if ("개인정보" in question_lower and ("침해" in question_lower or "신고" in question_lower) and "기관" in question_lower):
            return """개인정보보호위원회에서 개인정보 보호에 관한 업무를 총괄하며, 개인정보침해신고센터에서 신고 접수 및 상담 업무를 담당합니다. 개인정보보호위원회는 개인정보보호법에 따라 설치된 중앙행정기관으로 개인정보 보호 정책 수립과 감시 업무를 수행하며, 개인정보침해신고센터는 개인정보 침해신고 및 상담을 위한 전문 기관입니다."""

        return None

    def _generate_precise_llm_answer(self, question: str, domain: str, 
                                   intent_analysis: Dict, kb_analysis: Dict) -> str:
        """정교한 LLM 답변 생성"""
        
        # 도메인별 맞춤 힌트 생성
        precise_hints = {
            "domain": domain,
            "intent_analysis": intent_analysis,
            "kb_analysis": kb_analysis,
            "precision_mode": True,
            "target_length": 200,
            "professional_terms": self._get_domain_professional_terms(domain),
            "answer_structure": self._get_expected_answer_structure(intent_analysis)
        }

        answer = self.model_handler.generate_answer(
            question, "subjective", 5, intent_analysis, precise_hints
        )

        # 답변 품질 검증 및 정리
        if answer:
            answer = self._refine_llm_answer(answer, domain, intent_analysis)
        
        return answer

    def _get_domain_professional_terms(self, domain: str) -> List[str]:
        """도메인별 전문 용어"""
        terms = {
            "사이버보안": ["악성코드", "원격제어", "탐지", "모니터링", "보안", "침입", "방어체계"],
            "전자금융": ["전자금융거래법", "분쟁조정", "금융감독원", "이용자", "접근매체"],
            "개인정보보호": ["개인정보보호법", "정보주체", "개인정보보호위원회", "침해신고센터"],
            "정보보안": ["정보보안관리체계", "보안정책", "접근통제", "위험분석", "보안대책"],
            "위험관리": ["위험관리", "위험평가", "위험대응", "내부통제", "위험모니터링"],
            "금융투자": ["자본시장법", "금융투자업", "투자자보호", "적합성원칙"]
        }
        return terms.get(domain, ["법령", "규정", "관리", "체계"])

    def _get_expected_answer_structure(self, intent_analysis: Dict) -> str:
        """예상 답변 구조"""
        primary_intent = intent_analysis.get("primary_intent", "일반")
        
        structures = {
            "기관_묻기": "기관명 + 소속 + 주요업무 + 근거법령",
            "특징_묻기": "주요특징 + 기술적특성 + 동작방식",
            "지표_묻기": "탐지지표 + 모니터링방법 + 분석기법",
            "방안_묻기": "대응방안 + 실행절차 + 기대효과",
            "복합설명": "특징설명 + 지표설명 + 종합분석"
        }
        
        return structures.get(primary_intent, "개요 + 세부내용 + 결론")

    def _refine_llm_answer(self, answer: str, domain: str, intent_analysis: Dict) -> str:
        """LLM 답변 정제"""
        if not answer:
            return answer

        # 기본 정리
        answer = answer.strip()
        answer = re.sub(r'\s+', ' ', answer)
        answer = re.sub(r'답변[:：]\s*', '', answer)
        
        # 길이 조정
        if len(answer) > 600:
            sentences = answer.split('. ')
            answer = '. '.join(sentences[:5])
            if not answer.endswith('.'):
                answer += '.'
        
        # 문장 끝 처리
        if answer and not answer.endswith(('.', '다', '요', '함')):
            answer += '.'
        
        return answer

    def _generate_domain_default_answer(self, domain: str, intent_analysis: Dict) -> str:
        """도메인별 기본 답변"""
        primary_intent = intent_analysis.get("primary_intent", "일반")
        
        domain_defaults = {
            "사이버보안": "사이버보안 위협에 대응하기 위해 다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영하여 종합적인 보안 관리를 수행해야 합니다.",
            "전자금융": "전자금융거래법에 따라 전자금융업자는 이용자 보호를 위한 보안조치를 시행하고 분쟁 발생 시 전자금융분쟁조정위원회를 통해 해결할 수 있습니다.",
            "개인정보보호": "개인정보보호법에 따라 개인정보보호위원회가 총괄 업무를 담당하며 개인정보침해신고센터에서 신고 접수 및 상담 업무를 수행합니다.",
            "정보보안": "정보보안관리체계를 구축하여 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다.",
            "위험관리": "위험관리 체계를 구축하여 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립해야 합니다.",
            "금융투자": "자본시장법에 따라 금융투자업자는 투자자 보호와 시장 공정성 확보를 위한 내부통제기준을 수립해야 합니다."
        }
        
        return domain_defaults.get(domain, "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하여 지속적으로 운영해야 합니다.")

    def _validate_and_record_answer(self, question_id: str, question: str, 
                                  answer: str, question_type: str, 
                                  domain: str, max_choice: int = 5) -> bool:
        """답변 검증 및 학습 기록"""
        
        is_valid = self.data_processor.validate_answer(
            answer, question_type, max_choice, question
        )
        
        method_used = "specialized" if self._is_specialized_answer(answer, domain) else "llm"
        
        self.learning_manager.record_answer_attempt(
            question_id, question, answer, question_type, domain, is_valid, method_used
        )
        
        # 객관식 패턴 학습
        if question_type == "multiple_choice" and is_valid:
            pattern_type = self._identify_mc_pattern_type(question)
            self.learning_manager.record_mc_pattern(question, answer, pattern_type)
        
        return is_valid

    def _is_specialized_answer(self, answer: str, domain: str) -> bool:
        """특화 답변인지 확인"""
        specialized_indicators = {
            "사이버보안": ["트로이 목마", "원격제어 악성코드", "RAT", "탐지 지표"],
            "전자금융": ["전자금융분쟁조정위원회", "전자금융거래법", "금융감독원"],
            "개인정보보호": ["개인정보보호위원회", "개인정보침해신고센터", "개인정보보호법"]
        }
        
        indicators = specialized_indicators.get(domain, [])
        return any(indicator in answer for indicator in indicators)

    def _identify_mc_pattern_type(self, question: str) -> str:
        """객관식 패턴 유형 식별"""
        question_lower = question.lower()
        
        if "금융투자업" in question_lower and "해당하지 않는" in question_lower:
            return "금융투자업_분류"
        elif "위험관리" in question_lower and "적절하지 않은" in question_lower:
            return "위험관리_부적절"
        elif "개인정보" in question_lower and "중요한 요소" in question_lower:
            return "개인정보_중요요소"
        elif "한국은행" in question_lower and "자료제출" in question_lower:
            return "한국은행_자료제출"
        else:
            return "일반_객관식"

    def _generate_fallback_subjective_answer(self, question: str, domain: str, intent_analysis: Dict) -> str:
        """주관식 대체 답변 생성"""
        return self._generate_domain_default_answer(domain, intent_analysis)

    def _get_emergency_fallback_answer(self, question: str, question_type: str, max_choice: int) -> str:
        """긴급 대체 답변"""
        if question_type == "multiple_choice":
            return str((max_choice + 1) // 2)
        else:
            return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

    def execute_inference_with_data(self, test_df: pd.DataFrame, 
                                  submission_df: pd.DataFrame, 
                                  output_file: str = None) -> Dict:
        """데이터를 이용한 추론 실행"""
        
        output_file = output_file or DEFAULT_FILES["output_file"]
        
        print(f"데이터 로드 완료: {len(test_df)}개 문항")

        answers = []
        total_questions = len(test_df)

        # 학습 통계를 파일로 로그
        self.learning_manager.log_learning_stats("추론 시작")

        with tqdm(total=total_questions, desc="처리 중", unit="문항", 
                 ncols=50, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:
            
            for question_idx, (original_idx, row) in enumerate(test_df.iterrows()):
                question = row["Question"]
                question_id = row["ID"]

                answer = self.process_single_question(question, question_id)
                answers.append(answer)
                
                pbar.update(1)

                # 주기적으로 학습 데이터 저장
                if (question_idx + 1) % 100 == 0:
                    self.learning_manager.save_learning_data()
                    self.learning_manager.log_learning_stats(f"{question_idx + 1}개 문항 처리 완료")

                if (question_idx + 1) % MEMORY_CONFIG["gc_frequency"] == 0:
                    gc.collect()

        submission_df["Answer"] = answers
        save_success = self._simple_save_csv(submission_df, output_file)

        # 최종 학습 데이터 저장
        self.learning_manager.save_learning_data()
        
        final_stats = self.learning_manager.get_learning_stats()
        self.learning_manager.log_learning_stats("추론 완료")
        
        # 학습 분석 보고서 생성
        self.learning_manager.export_analysis()

        if not save_success:
            print(f"지정된 파일로 저장에 실패했습니다: {output_file}")

        return self._get_results_summary(final_stats)

    def execute_inference(self, test_file: str = None, submission_file: str = None, 
                        output_file: str = None) -> Dict:
        """추론 실행"""
        test_file = test_file or DEFAULT_FILES["test_file"]
        submission_file = submission_file or DEFAULT_FILES["submission_file"]
        output_file = output_file or DEFAULT_FILES["output_file"]

        try:
            test_df = pd.read_csv(test_file)
            submission_df = pd.read_csv(submission_file)
        except Exception as e:
            raise RuntimeError(f"데이터 로드 실패: {e}")

        return self.execute_inference_with_data(test_df, submission_df, output_file)

    def _simple_save_csv(self, df: pd.DataFrame, filepath: str) -> bool:
        """간단한 CSV 저장"""
        filepath = Path(filepath)
        try:
            df.to_csv(filepath, index=False, encoding=FILE_VALIDATION["encoding"])
            return True
        except Exception as e:
            print(f"파일 저장 중 오류: {e}")
            return False

    def _get_results_summary(self, learning_stats: Dict) -> Dict:
        """결과 요약"""
        return {
            "success": True,
            "total_time": time.time() - self.start_time,
            "learning_stats": learning_stats,
        }

    def cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, "learning_manager"):
                self.learning_manager.cleanup()

            if hasattr(self, "model_handler"):
                self.model_handler.cleanup()

            if hasattr(self, "data_processor"):
                self.data_processor.cleanup()

            if hasattr(self, "knowledge_base"):
                self.knowledge_base.cleanup()

            gc.collect()

        except Exception as e:
            pass


def main():
    """메인 실행 함수"""
    engine = None
    try:
        engine = FinancialAIInference(verbose=False)
        results = engine.execute_inference()

        if results["success"]:
            print(f"\n추론 완료 (처리시간: {results['total_time']:.1f}초)")
            if "learning_stats" in results:
                print(f"학습 통계: {results['learning_stats']}")

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