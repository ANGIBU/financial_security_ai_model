# inference.py
"""
실행 파일
"""

import os
import sys
import time
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
from typing import List, Dict, Tuple, Optional
import signal
import psutil
warnings.filterwarnings("ignore")

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from model_handler import OptimizedModelHandler
from data_processor import IntelligentDataProcessor
from prompt_engineering import AdvancedPromptEngineer
from knowledge_base import FinancialSecurityKnowledgeBase
from advanced_optimizer import AdvancedOptimizer, ResponseValidator, PerformanceMonitor

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("작업 시간 초과")

class HighPerformanceInferenceEngine:
    """추론 엔진"""
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.start_time = time.time()
        self.time_limit = 4.25 * 3600  # 4시간 15분 (여유 15분)
        
        print("시스템 초기화 중...")
        
        # 핵심 컴포넌트
        self.model_handler = OptimizedModelHandler(**model_config)
        self.data_processor = IntelligentDataProcessor()
        self.prompt_engineer = AdvancedPromptEngineer()
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        
        # 고급 최적화 컴포넌트
        self.optimizer = AdvancedOptimizer()
        self.validator = ResponseValidator()
        self.monitor = PerformanceMonitor()
        
        # 성능 추적
        self.performance_stats = {
            "total_questions": 0,
            "successful": 0,
            "failed": 0,
            "cache_hits": 0,
            "batch_success": 0,
            "confidence_sum": 0
        }
        
        print("초기화 완료")
    
    def execute_inference(self, test_file: str, submission_file: str, 
                         output_file: str = "./final_submission.csv") -> Dict:
        """메인 추론 실행 - 고급 최적화"""
        
        # 데이터 로드
        test_df, sample_submission = self._load_data(test_file, submission_file)
        questions = test_df['Question'].tolist()
        question_ids = test_df['ID'].tolist()
        
        # 고급 문제 분석
        analyzed_questions = self._advanced_question_analysis(questions)
        
        print(f"\n=== 문제 분석 완료 ===")
        print(f"총 문항: {len(questions)}")
        print(f"쉬움: {sum(1 for q in analyzed_questions if q['difficulty'].score < 0.3)}개")
        print(f"보통: {sum(1 for q in analyzed_questions if 0.3 <= q['difficulty'].score < 0.6)}개")
        print(f"어려움: {sum(1 for q in analyzed_questions if q['difficulty'].score >= 0.6)}개")
        
        # 우선순위 재정렬
        prioritized_questions = self.optimizer.prioritize_questions(analyzed_questions)
        
        # 추론 실행
        predictions = [""] * len(questions)
        self._execute_adaptive_inference(prioritized_questions, questions, predictions)
        
        # 결과 검증 및 개선
        self._validate_and_improve_predictions(predictions, questions)
        
        # 결과 저장
        results = self._save_results(predictions, sample_submission, output_file)
        
        return results
    
    def _load_data(self, test_file: str, submission_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 로드"""
        try:
            test_df = pd.read_csv(test_file)
            sample_submission = pd.read_csv(submission_file)
            
            print(f"데이터 로드 완료: {len(test_df)}개 문항")
            return test_df, sample_submission
            
        except Exception as e:
            print(f"데이터 로딩 오류: {e}")
            sys.exit(1)
    
    def _advanced_question_analysis(self, questions: List[str]) -> List[Dict]:
        """고급 문제 분석"""
        analyzed = []
        
        print("고급 문제 분석 중...")
        for idx, question in enumerate(tqdm(questions, desc="분석")):
            # 구조 분석
            structure = self.data_processor.analyze_question_structure(question)
            
            # 난이도 평가
            difficulty = self.optimizer.evaluate_question_difficulty(question, structure)
            
            # 지식베이스 분석
            kb_analysis = self.knowledge_base.analyze_question(question)
            
            # 답변 힌트
            hint_answer, hint_confidence = self.optimizer.get_smart_answer_hint(question, structure)
            
            analyzed.append({
                "index": idx,
                "question": question,
                "type": structure["question_type"],
                "structure": structure,
                "difficulty": difficulty,
                "kb_analysis": kb_analysis,
                "hint_answer": hint_answer,
                "hint_confidence": hint_confidence,
                "priority_score": 0  # 나중에 설정
            })
        
        return analyzed
    
    def _execute_adaptive_inference(self, prioritized_questions: List[Dict],
                                  original_questions: List[str],
                                  predictions: List[str]):
        """적응형 추론 실행"""
        
        # 배치 처리 가능한 쉬운 문제들
        easy_mc = [q for q in prioritized_questions 
                  if q["type"] == "multiple_choice" and q["difficulty"].score < 0.4]
        
        # 개별 처리 필요한 문제들
        complex_questions = [q for q in prioritized_questions 
                           if q not in easy_mc]
        
        print(f"\n배치 처리: {len(easy_mc)}개, 개별 처리: {len(complex_questions)}개")
        
        # 1단계: 쉬운 문제 배치 처리
        if easy_mc:
            self._process_easy_batch_advanced(easy_mc, predictions)
        
        # 2단계: 복잡한 문제 개별 처리
        self._process_complex_adaptive(complex_questions, original_questions, predictions)
        
        # 3단계: 미처리 문제 긴급 처리
        self._emergency_process_remaining(predictions, original_questions)
    
    def _process_easy_batch_advanced(self, easy_questions: List[Dict],
                                   predictions: List[str]):
        """쉬운 문제 고급 배치 처리"""
        
        # 동적 배치 크기
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        question_lengths = [len(q["question"]) for q in easy_questions]
        batch_size = self.optimizer.optimize_batch_size(available_memory, question_lengths)
        
        print(f"동적 배치 크기: {batch_size}")
        
        with tqdm(total=len(easy_questions), desc="배치 처리") as pbar:
            for i in range(0, len(easy_questions), batch_size):
                batch = easy_questions[i:i+batch_size]
                
                # 높은 확신도 힌트가 있는 경우 바로 사용
                for q in batch:
                    if q["hint_confidence"] > 0.7:
                        predictions[q["index"]] = q["hint_answer"]
                        self.performance_stats["cache_hits"] += 1
                        pbar.update(1)
                        continue
                
                # 나머지는 배치 추론
                remaining_batch = [q for q in batch if predictions[q["index"]] == ""]
                if remaining_batch:
                    try:
                        prompts = []
                        for q in remaining_batch:
                            prompt = self.prompt_engineer.create_simple_mc_prompt(q["question"])
                            optimized = self.prompt_engineer.optimize_for_model(
                                prompt, self.model_config["model_name"]
                            )
                            prompts.append(optimized)
                        
                        # 배치 추론
                        results = self.model_handler.generate_batch_responses(
                            prompts,
                            ["multiple_choice"] * len(prompts),
                            batch_size=len(prompts)
                        )
                        
                        # 결과 저장
                        for q, result in zip(remaining_batch, results):
                            answer = self.data_processor.extract_mc_answer_fast(result.response)
                            predictions[q["index"]] = answer
                            self.performance_stats["batch_success"] += 1
                            self.monitor.update(result.inference_time, result.confidence)
                        
                        pbar.update(len(remaining_batch))
                        
                    except Exception as e:
                        # 실패 시 힌트 사용
                        for q in remaining_batch:
                            predictions[q["index"]] = q["hint_answer"]
                        pbar.update(len(remaining_batch))
                
                # 메모리 정리
                if i % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
    
    def _process_complex_adaptive(self, complex_questions: List[Dict],
                                original_questions: List[str],
                                predictions: List[str]):
        """복잡한 문제 적응형 처리"""
        
        with tqdm(total=len(complex_questions), desc="개별 처리") as pbar:
            for q_info in complex_questions:
                idx = q_info["index"]
                
                # 이미 처리된 경우 스킵
                if predictions[idx] != "":
                    pbar.update(1)
                    continue
                
                # 적응형 타임아웃
                timeout = self.monitor.get_adaptive_timeout()
                timeout = min(timeout, q_info["difficulty"].recommended_time)
                
                # 재시도 여부 결정
                max_attempts = 1 if self.monitor.should_skip_retries() else \
                             q_info["difficulty"].recommended_attempts
                
                try:
                    # 타임아웃 설정
                    signal.alarm(int(timeout))
                    
                    # 전략 선택
                    if q_info["kb_analysis"].get("relevant_laws"):
                        strategy = "law_focused"
                    elif q_info["structure"].get("has_negative"):
                        strategy = "negative_specialized"
                    else:
                        strategy = "balanced"
                    
                    # 프롬프트 생성
                    prompt = self.prompt_engineer.create_expert_prompt(
                        q_info["question"], q_info["type"], strategy
                    )
                    optimized_prompt = self.prompt_engineer.optimize_for_model(
                        prompt, self.model_config["model_name"]
                    )
                    
                    # 추론 실행
                    start_time = time.time()
                    result = self.model_handler.generate_expert_response(
                        optimized_prompt,
                        q_info["type"],
                        max_attempts=max_attempts
                    )
                    inference_time = time.time() - start_time
                    
                    signal.alarm(0)  # 타임아웃 해제
                    
                    # 응답 검증
                    is_valid, issues = self.validator.validate_response(
                        result.response, q_info["type"]
                    )
                    
                    if not is_valid:
                        # 응답 개선
                        improved = self.validator.improve_response(
                            result.response, issues, q_info["type"]
                        )
                        final_answer = improved
                    else:
                        # 후처리
                        final_answer = self.data_processor.post_process_answer(
                            result.response, q_info["question"], q_info["type"]
                        )
                    
                    predictions[idx] = final_answer
                    self.performance_stats["successful"] += 1
                    self.monitor.update(inference_time, result.confidence)
                    
                except TimeoutException:
                    signal.alarm(0)
                    # 힌트 사용
                    predictions[idx] = q_info["hint_answer"] if q_info["type"] == "multiple_choice" \
                                     else "해당 사항에 대한 전문적 검토가 필요합니다."
                    self.performance_stats["failed"] += 1
                    
                except Exception as e:
                    signal.alarm(0)
                    predictions[idx] = self._get_emergency_answer(q_info)
                    self.performance_stats["failed"] += 1
                
                pbar.update(1)
    
    def _emergency_process_remaining(self, predictions: List[str], 
                                   questions: List[str]):
        """미처리 문제 긴급 처리"""
        
        remaining = predictions.count("")
        if remaining == 0:
            return
        
        print(f"\n긴급 처리 모드: {remaining}개 남음")
        
        for i, pred in enumerate(predictions):
            if pred == "":
                structure = self.data_processor.analyze_question_structure(questions[i])
                hint_answer, _ = self.optimizer.get_smart_answer_hint(questions[i], structure)
                
                if structure["question_type"] == "multiple_choice":
                    predictions[i] = hint_answer
                else:
                    predictions[i] = "금융보안 정책에 따른 체계적 관리가 필요합니다."
    
    def _validate_and_improve_predictions(self, predictions: List[str], 
                                        questions: List[str]):
        """예측 결과 검증 및 개선"""
        
        print("\n최종 검증 중...")
        improved_count = 0
        
        for i, (pred, question) in enumerate(zip(predictions, questions)):
            structure = self.data_processor.analyze_question_structure(question)
            is_valid, issues = self.validator.validate_response(pred, structure["question_type"])
            
            if not is_valid:
                improved = self.validator.improve_response(pred, issues, structure["question_type"])
                predictions[i] = improved
                improved_count += 1
        
        if improved_count > 0:
            print(f"개선된 답변: {improved_count}개")
    
    def _get_emergency_answer(self, q_info: Dict) -> str:
        """긴급 답변 생성"""
        if q_info["type"] == "multiple_choice":
            return q_info["hint_answer"]
        else:
            domain = q_info["kb_analysis"].get("domain", ["일반"])[0]
            if domain == "개인정보보호":
                return "개인정보보호법에 따른 안전성 확보조치가 필요합니다."
            elif domain == "전자금융":
                return "전자금융거래법에 따른 보안 대책 수립이 필요합니다."
            else:
                return "금융보안 규정에 따른 종합적인 대책이 필요합니다."
    
    def _save_results(self, predictions: List[str], 
                    sample_submission: pd.DataFrame, 
                    output_file: str) -> Dict:
        """결과 저장 및 분석"""
        
        try:
            # 결과 저장
            sample_submission['Answer'] = predictions
            sample_submission.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 통계 계산
            total_time = time.time() - self.start_time
            mc_answers = [p for p in predictions if p.strip().isdigit()]
            
            # 분포 계산
            distribution = {}
            for answer in mc_answers:
                distribution[answer] = distribution.get(answer, 0) + 1
            
            results = {
                "output_file": output_file,
                "total_questions": len(predictions),
                "total_time_minutes": total_time / 60,
                "successful": self.performance_stats["successful"],
                "failed": self.performance_stats["failed"],
                "cache_hits": self.performance_stats["cache_hits"],
                "batch_success": self.performance_stats["batch_success"],
                "answer_distribution": distribution,
                "avg_confidence": self.monitor.stats["avg_confidence"],
                "success": True
            }
            
            # 결과 출력
            print("\n=== 최종 결과 ===")
            print(f"총 처리: {len(predictions)}개")
            print(f"소요 시간: {total_time/60:.1f}분")
            print(f"성공: {self.performance_stats['successful']}개")
            print(f"캐시 히트: {self.performance_stats['cache_hits']}개")
            print(f"배치 성공: {self.performance_stats['batch_success']}개")
            print(f"평균 신뢰도: {self.monitor.stats['avg_confidence']:.3f}")
            
            if mc_answers:
                print("\n객관식 답변 분포:")
                for choice in sorted(distribution.keys()):
                    count = distribution[choice]
                    pct = (count / len(mc_answers)) * 100
                    print(f"  {choice}번: {count}개 ({pct:.1f}%)")
            
            print(f"\n최종 제출 파일: {output_file}")
            
            return results
            
        except Exception as e:
            print(f"결과 저장 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            print(f"\n캐시 히트율: {self.model_handler.cache_hits}회")
            self.model_handler.cleanup()
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"정리 중 오류: {e}")

# 시그널 핸들러 설정
signal.signal(signal.SIGALRM, timeout_handler)

def main():
    """메인 함수"""
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("오류: CUDA 사용 불가")
        sys.exit(1)
    
    gpu_info = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_info.name} ({gpu_info.total_memory / (1024**3):.1f}GB)")
    
    # 파일 확인
    test_file = "./test.csv"
    submission_file = "./sample_submission.csv"
    
    if not os.path.exists(test_file) or not os.path.exists(submission_file):
        print("오류: 데이터 파일 없음")
        sys.exit(1)
    
    # 모델 설정 (RTX 4090 24GB)
    model_config = {
        "model_name": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "device": "cuda",
        "load_in_4bit": False,
        "max_memory_gb": 22
    }
    
    # 추론 실행
    engine = None
    try:
        engine = HighPerformanceInferenceEngine(model_config)
        results = engine.execute_inference(test_file, submission_file)
        
        if results["success"]:
            print("\n✅ 추론 성공적으로 완료!")
            print(f"평균 신뢰도: {results['avg_confidence']:.3f}")
        
    except KeyboardInterrupt:
        print("\n추론 중단")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.cleanup()

if __name__ == "__main__":
    main()