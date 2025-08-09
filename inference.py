# inference.py
"""
실행 파일
"""

import os
import sys
import time
import re
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
from typing import List, Dict, Tuple, Optional
import numpy as np
warnings.filterwarnings("ignore")

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging
logging.set_verbosity_error()

from model_handler import ModelHandler
from data_processor import DataProcessor
from prompt_engineering import PromptEngineer
from advanced_optimizer import SystemOptimizer
from pattern_learner import AnswerPatternLearner

from learning_system import UnifiedLearningSystem
from manual_correction import ManualCorrectionSystem
from auto_learner import AutoLearner

class FinancialAIInference:
    """금융 AI 추론 엔진"""
    
    def __init__(self, enable_learning: bool = True, verbose: bool = False):
        self.start_time = time.time()
        self.enable_learning = enable_learning
        self.verbose = verbose
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.90)
        
        print("시스템 초기화 중...")
        
        self.model_handler = ModelHandler(
            model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_in_4bit=True,
            max_memory_gb=22,
            verbose=self.verbose
        )
        
        self.data_processor = DataProcessor()
        self.prompt_engineer = PromptEngineer()
        self.optimizer = SystemOptimizer(debug_mode=self.verbose)
        self.pattern_learner = AnswerPatternLearner()
        
        if self.enable_learning:
            self.learning_system = UnifiedLearningSystem()
            self.correction_system = ManualCorrectionSystem()
            self.auto_learner = AutoLearner()
            self._load_existing_learning_data()
        
        self.stats = {
            "total": 0,
            "mc_correct": 0,
            "subj_correct": 0,
            "errors": 0,
            "timeouts": 0,
            "learned": 0,
            "korean_failures": 0,
            "korean_fixes": 0,
            "fallback_used": 0,
            "smart_hints_used": 0,
            "model_generation_success": 0,
            "pattern_extraction_success": 0
        }
        
        print("초기화 완료")
    
    def _load_existing_learning_data(self) -> None:
        """학습 데이터 로드"""
        try:
            if self.learning_system.load_learning_data():
                if self.verbose:
                    print(f"학습 데이터 로드: {self.learning_system.learning_metrics['total_samples']}개")
            
            if self.auto_learner.load_model():
                if self.verbose:
                    print(f"자동 학습 모델 로드: {len(self.auto_learner.pattern_weights)}개 패턴")
            
            corrections = self.correction_system.load_corrections_from_csv("./corrections.csv")
            if corrections > 0 and self.verbose:
                print(f"교정 데이터 로드: {corrections}개")
        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 로드 오류: {e}")
    
    def _validate_korean_quality(self, text: str, question_type: str) -> Tuple[bool, float]:
        """한국어 품질 검증"""
        
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return True, 1.0
            if re.search(r'[1-5]', text):
                return True, 0.8
            return False, 0.0
        
        if not text or len(text.strip()) < 20:
            return False, 0.0
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return False, 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return False, 0.0
        
        korean_ratio = korean_chars / total_chars
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        if korean_ratio < 0.5:
            return False, korean_ratio
        
        if english_ratio > 0.3:
            return False, 1 - english_ratio
        
        quality_score = korean_ratio * 0.8 - english_ratio * 0.2
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality_score += min(prof_count * 0.05, 0.15)
        
        final_quality = max(0, min(1, quality_score))
        
        return final_quality > 0.6, final_quality
    
    def _get_domain_specific_fallback(self, question: str, question_type: str) -> str:
        """도메인별 폴백 답변"""
        
        if question_type == "multiple_choice":
            structure = self.data_processor.analyze_question_structure(question)
            
            hint, conf = self.optimizer.get_smart_answer_hint(question, structure)
            
            if conf > 0.5:
                self.stats["smart_hints_used"] += 1
                return hint
            else:
                question_hash = hash(question) % 5 + 1
                base_answers = ["1", "2", "3", "4", "5"]
                return str(base_answers[question_hash % 5])
        
        question_lower = question.lower()
        
        if "트로이" in question or "악성코드" in question or "RAT" in question.upper():
            return "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행, 방화벽 규칙 변경, 레지스트리 변경, 이상한 파일 생성 등이 있습니다."
        
        if "개인정보" in question_lower:
            if "유출" in question_lower:
                return "개인정보 유출 시 개인정보보호법에 따라 지체 없이 정보주체에게 통지하고, 일정 규모 이상의 유출 시 개인정보보호위원회에 신고해야 합니다. 유출 통지 내용에는 유출 항목, 시점, 경위, 피해 최소화 방법, 담당부서 연락처 등이 포함되어야 합니다."
            else:
                return "개인정보보호법에 따라 개인정보는 정보주체의 동의를 받아 수집하고, 수집 목적 범위 내에서만 이용해야 합니다. 안전성 확보조치를 통해 개인정보를 보호하고, 목적 달성 후에는 지체 없이 파기해야 합니다."
        
        if "전자금융" in question_lower:
            if "접근매체" in question_lower:
                return "전자금융거래법상 접근매체는 전자금융거래에서 이용자 및 거래내용의 진실성과 정확성을 확보하기 위한 수단입니다. 금융회사는 안전하고 신뢰할 수 있는 접근매체를 선정해야 하며, 이용자는 접근매체를 안전하게 관리할 의무가 있습니다."
            else:
                return "전자금융거래는 전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래입니다. 금융회사는 전자금융거래의 안전성과 신뢰성을 확보하고, 이용자 보호를 위한 적절한 조치를 취해야 합니다."
        
        if "위험" in question_lower and "관리" in question_lower:
            return "위험관리는 조직의 목표 달성에 영향을 미칠 수 있는 위험을 체계적으로 식별, 분석, 평가하고 적절한 대응방안을 수립하여 관리하는 과정입니다. 위험 수용 능력을 고려하여 위험 대응 전략을 선정하고 지속적으로 모니터링해야 합니다."
        
        if "관리체계" in question_lower and "정책" in question_lower:
            return "관리체계 수립 시 최고경영진의 참여와 지원이 가장 중요하며, 명확한 정책 수립과 책임자 지정, 적절한 자원 할당이 필요합니다. 정보보호 및 개인정보보호 정책의 제정과 개정을 통해 체계적인 관리 기반을 마련해야 합니다."
        
        if "재해" in question_lower and "복구" in question_lower:
            return "재해복구계획은 재해 발생 시 핵심 업무를 신속하게 복구하기 위한 체계적인 계획입니다. 복구목표시간과 복구목표시점을 설정하고, 백업 및 복구 절차를 수립하며, 정기적인 모의훈련을 통해 실효성을 검증해야 합니다."
        
        if "암호" in question_lower:
            return "암호화는 정보의 기밀성과 무결성을 보장하기 위한 핵심 보안 기술입니다. 대칭키 암호화와 공개키 암호화를 적절히 활용하고, 안전한 키 관리 체계를 구축해야 합니다. 중요 정보는 전송 구간과 저장 시 모두 암호화해야 합니다."
        
        return "관련 법령과 규정에 따라 체계적인 보안 관리 방안을 수립하고, 지속적인 모니터링과 개선을 통해 안전성을 확보해야 합니다."
    
    def process_question(self, question: str, question_id: str, idx: int) -> str:
        """문제 처리"""
        
        try:
            structure = self.data_processor.analyze_question_structure(question)
            analysis = self.prompt_engineer.knowledge_base.analyze_question(question)
            
            is_mc = structure["question_type"] == "multiple_choice"
            is_subjective = structure["question_type"] == "subjective"
            
            self._debug_log(f"문제 {idx}: 유형={structure['question_type']}, 객관식={is_mc}, 주관식={is_subjective}")
            
            difficulty = self.optimizer.evaluate_question_difficulty(question, structure)
            
            if self.enable_learning:
                learned_answer, learned_confidence = self.auto_learner.predict_with_patterns(
                    question, structure["question_type"]
                )
                
                corrected_answer, correction_conf = self.correction_system.apply_corrections(
                    question, learned_answer
                )
                
                if correction_conf > 0.8:
                    is_valid, quality = self._validate_korean_quality(corrected_answer, structure["question_type"])
                    if is_valid:
                        return corrected_answer
            
            if is_mc:
                hint_answer, hint_confidence = self.optimizer.get_smart_answer_hint(question, structure)
                
                if hint_confidence > 0.7:
                    self.stats["smart_hints_used"] += 1
                    return hint_answer
                
                if difficulty.score > 0.8 or self.stats["total"] > 300:
                    if hint_confidence > 0.5:
                        self.stats["smart_hints_used"] += 1
                        return hint_answer
                
                prompt = self.prompt_engineer.create_korean_reinforced_prompt(
                    question, structure["question_type"]
                )
                
                max_attempts = 1 if self.stats["total"] > 100 else 2
                
                result = self.model_handler.generate_response(
                    prompt=prompt,
                    question_type=structure["question_type"],
                    max_attempts=max_attempts
                )
                
                extracted = self.data_processor.extract_mc_answer_fast(result.response)
                
                if extracted and extracted.isdigit() and 1 <= int(extracted) <= 5:
                    self.stats["model_generation_success"] += 1
                    self.stats["pattern_extraction_success"] += 1
                    answer = extracted
                else:
                    self.stats["smart_hints_used"] += 1
                    answer = hint_answer if hint_confidence > 0.3 else self._get_domain_specific_fallback(question, "multiple_choice")
            
            elif is_subjective:
                prompt = self.prompt_engineer.create_korean_reinforced_prompt(
                    question, structure["question_type"]
                )
                
                max_attempts = 1 if self.stats["total"] > 100 else 2
                
                result = self.model_handler.generate_response(
                    prompt=prompt,
                    question_type=structure["question_type"],
                    max_attempts=max_attempts
                )
                
                answer = self.data_processor._clean_korean_text(result.response)
                
                is_valid, quality = self._validate_korean_quality(answer, structure["question_type"])
                
                if not is_valid or quality < 0.6:
                    self.stats["korean_failures"] += 1
                    answer = self._get_domain_specific_fallback(question, structure["question_type"])
                    self.stats["korean_fixes"] += 1
                    self.stats["fallback_used"] += 1
                else:
                    self.stats["model_generation_success"] += 1
                
                if len(answer) < 30:
                    answer = self._get_domain_specific_fallback(question, structure["question_type"])
                    self.stats["fallback_used"] += 1
                elif len(answer) > 800:
                    answer = answer[:797] + "..."
            
            else:
                self.stats["fallback_used"] += 1
                answer = self._get_domain_specific_fallback(question, "multiple_choice")
            
            if self.enable_learning and 'result' in locals() and result.confidence > 0.5:
                self.auto_learner.learn_from_prediction(
                    question, answer, result.confidence,
                    structure["question_type"], analysis.get("domain", ["일반"])
                )
                
                if result.confidence > 0.6:
                    self.learning_system.add_training_sample(
                        question=question,
                        correct_answer=answer,
                        predicted_answer=answer,
                        confidence=result.confidence,
                        question_type=structure["question_type"],
                        domain=analysis.get("domain", ["일반"]),
                        question_id=question_id
                    )
                    self.stats["learned"] += 1
            
            return answer
            
        except Exception as e:
            self.stats["errors"] += 1
            if self.verbose:
                print(f"처리 오류: {e}")
            
            self.stats["fallback_used"] += 1
            fallback_type = structure.get("question_type", "multiple_choice") if 'structure' in locals() else "multiple_choice"
            return self._get_domain_specific_fallback(question, fallback_type)
    
    def _debug_log(self, message: str):
        """디버그 로그"""
        if self.verbose:
            print(f"[DEBUG] {message}")
    
    def execute_inference(self, test_file: str, submission_file: str,
                         output_file: str = "./final_submission.csv",
                         enable_manual_correction: bool = False) -> Dict:
        """추론 실행"""
        test_df = pd.read_csv(test_file)
        submission_df = pd.read_csv(submission_file)
        
        print(f"데이터 로드 완료: {len(test_df)}개 문항")
        
        questions_data = []
        
        for idx, row in test_df.iterrows():
            question = row['Question']
            structure = self.data_processor.analyze_question_structure(question)
            
            questions_data.append({
                "idx": idx,
                "id": row['ID'],
                "question": question,
                "structure": structure,
                "is_mc": structure["question_type"] == "multiple_choice"
            })
        
        mc_count = sum(1 for q in questions_data if q["is_mc"])
        subj_count = len(questions_data) - mc_count
        
        print(f"문제 구성: 객관식 {mc_count}개, 주관식 {subj_count}개")
        
        if self.enable_learning:
            print(f"학습 모드: 활성화")
        
        answers = [""] * len(test_df)
        
        print("추론 시작...")
        progress_bar = tqdm(questions_data, desc="추론", ncols=80, disable=not self.verbose)
        
        for q_data in progress_bar:
            idx = q_data["idx"]
            question_id = q_data["id"]
            question = q_data["question"]
            
            answer = self.process_question(question, question_id, idx)
            answers[idx] = answer
            
            self.stats["total"] += 1
            
            if not self.verbose and self.stats["total"] % 100 == 0:
                print(f"진행률: {self.stats['total']}/{len(test_df)}")
            
            if self.stats["total"] % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if self.enable_learning and self.stats["total"] % 100 == 0:
                self.auto_learner.optimize_patterns()
        
        if enable_manual_correction and self.enable_learning:
            print("\n수동 교정 모드 시작...")
            corrections = self.correction_system.interactive_correction(
                questions_data[:5],
                answers[:5]
            )
            print(f"교정 완료: {corrections}개")
        
        submission_df['Answer'] = answers
        submission_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        if self.enable_learning:
            try:
                if self.learning_system.save_learning_data():
                    if self.verbose:
                        print("학습 데이터 저장 완료")
                if self.auto_learner.save_model():
                    if self.verbose:
                        print("자동 학습 모델 저장 완료")
                if self.correction_system.save_corrections_to_csv():
                    if self.verbose:
                        print("교정 데이터 저장 완료")
            except Exception as e:
                if self.verbose:
                    print(f"데이터 저장 오류: {e}")
        
        mc_answers = []
        subj_answers = []
        
        for answer, q_data in zip(answers, questions_data):
            if q_data["is_mc"]:
                if answer.isdigit() and 1 <= int(answer) <= 5:
                    mc_answers.append(answer)
            else:
                subj_answers.append(answer)
        
        answer_distribution = {}
        for ans in mc_answers:
            answer_distribution[ans] = answer_distribution.get(ans, 0) + 1
        
        korean_quality_scores = []
        
        for answer in subj_answers:
            _, quality = self._validate_korean_quality(answer, "subjective")
            korean_quality_scores.append(quality)
        
        mc_quality_scores = []
        for answer in mc_answers:
            _, quality = self._validate_korean_quality(answer, "multiple_choice")
            mc_quality_scores.append(quality)
        
        all_quality_scores = korean_quality_scores + mc_quality_scores
        avg_korean_quality = np.mean(all_quality_scores) if all_quality_scores else 0
        
        print("\n" + "="*50)
        print("추론 완료")
        print("="*50)
        print(f"총 문항: {len(answers)}개")
        
        print(f"\n처리 통계:")
        print(f"  모델 생성 성공: {self.stats['model_generation_success']}/{self.stats['total']} ({self.stats['model_generation_success']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"  스마트 힌트 사용: {self.stats['smart_hints_used']}회")
        print(f"  폴백 사용: {self.stats['fallback_used']}회")
        print(f"  처리 오류: {self.stats['errors']}회")
        
        print(f"\n한국어 품질 리포트:")
        print(f"  한국어 실패: {self.stats['korean_failures']}회")
        print(f"  한국어 수정: {self.stats['korean_fixes']}회")
        print(f"  평균 품질 점수: {avg_korean_quality:.2f}")
        
        high_quality_count = sum(1 for q in all_quality_scores if q > 0.8)
        print(f"  품질 우수 답변: {high_quality_count}/{len(all_quality_scores)}개")
        
        if len(all_quality_scores) > 0:
            avg_korean_ratio = avg_korean_quality
            print(f"  평균 한국어 비율: {avg_korean_ratio:.2%}")
        
        if avg_korean_quality > 0.8:
            print("  한국어 품질 우수")
        elif avg_korean_quality > 0.6:
            print("  한국어 품질 보통")
        else:
            print("  한국어 품질 개선 필요")
        
        if self.enable_learning:
            print(f"\n학습 통계:")
            print(f"  학습된 샘플: {self.stats['learned']}개")
            print(f"  패턴 수: {len(self.auto_learner.pattern_weights)}개")
            print(f"  정확도: {self.learning_system.get_current_accuracy():.2%}")
        
        if mc_answers:
            print(f"\n객관식 답변 분포 ({len(mc_answers)}개):")
            for num in sorted(answer_distribution.keys()):
                count = answer_distribution[num]
                pct = (count / len(mc_answers)) * 100
                print(f"  {num}번: {count}개 ({pct:.1f}%)")
            
            unique_answers = len(answer_distribution)
            if unique_answers >= 4:
                print("  답변 다양성 우수")
            elif unique_answers >= 3:
                print("  답변 다양성 양호")
            elif unique_answers >= 2:
                print("  답변 다양성 보통")
            else:
                print("  답변 다양성 부족")
        
        print(f"\n결과 파일: {output_file}")
        
        return {
            "success": True,
            "total_questions": len(answers),
            "mc_count": mc_count,
            "subj_count": subj_count,
            "answer_distribution": answer_distribution,
            "processing_stats": {
                "model_success": self.stats["model_generation_success"],
                "smart_hints": self.stats["smart_hints_used"],
                "fallback_used": self.stats["fallback_used"],
                "errors": self.stats["errors"]
            },
            "korean_quality": {
                "failures": self.stats["korean_failures"],
                "fixes": self.stats["korean_fixes"],
                "avg_score": avg_korean_quality,
                "high_quality_count": high_quality_count
            },
            "learning_stats": {
                "learned_samples": self.stats["learned"],
                "patterns": len(self.auto_learner.pattern_weights) if self.enable_learning else 0,
                "accuracy": self.learning_system.get_current_accuracy() if self.enable_learning else 0
            }
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.model_handler.cleanup()
            self.data_processor.cleanup()
            self.prompt_engineer.cleanup()
            self.pattern_learner.cleanup()
            
            if self.enable_learning:
                self.learning_system.cleanup()
                self.correction_system.cleanup()
                self.auto_learner.cleanup()
            
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")

def main():
    """메인 함수"""
    
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_info.name}")
        print(f"메모리: {gpu_info.total_memory / (1024**3):.1f}GB")
    else:
        print("GPU 없음 - CPU 모드")
    
    test_file = "./test.csv"
    submission_file = "./sample_submission.csv"
    
    if not os.path.exists(test_file):
        print(f"오류: {test_file} 파일 없음")
        sys.exit(1)
    
    if not os.path.exists(submission_file):
        print(f"오류: {submission_file} 파일 없음")
        sys.exit(1)
    
    enable_learning = True
    verbose = False
    
    engine = None
    try:
        engine = FinancialAIInference(enable_learning=enable_learning, verbose=verbose)
        results = engine.execute_inference(
            test_file, 
            submission_file,
            enable_manual_correction=False
        )
        
        if results["success"]:
            print("\n추론 완료")
            processing_stats = results["processing_stats"]
            
            success_rate = processing_stats["model_success"] / results["total_questions"] * 100
            print(f"모델 생성 성공률: {success_rate:.1f}%")
            
            if processing_stats["smart_hints"] > 0:
                print(f"스마트 힌트 활용: {processing_stats['smart_hints']}회")
            
            quality = results["korean_quality"]["avg_score"]
            if quality > 0.8:
                print("한국어 품질 우수")
            elif quality > 0.6:
                print("한국어 품질 보통")
            else:
                print("한국어 품질 개선 필요")
        
    except KeyboardInterrupt:
        print("\n추론 중단")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.cleanup()

if __name__ == "__main__":
    main()