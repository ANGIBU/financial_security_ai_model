# test_runner.py

"""
통합 추론 테스트 실행기
- 50문항 딥러닝 테스트 실행
- 실제 GPU 추론 및 학습 시스템 연동
- 파인튜닝된 모델 지원
- 상세한 성능 검증 및 분석
- 논리적 추론 성능 측정
- CoT 추론 과정 검증
- 추론 품질 평가 메트릭
- 통합된 추론 파이프라인 성능 분석
- 실시간 진행상황 모니터링
- 딥러닝 학습 과정 추적
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import threading
import queue

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from inference import FinancialAIInference

DEFAULT_TEST_SIZE = 50
MAX_TEST_SIZE = 500
MIN_TEST_SIZE = 1

# 성능 모니터링 상수
PROGRESS_UPDATE_INTERVAL = 5
DETAILED_ANALYSIS_INTERVAL = 10
MEMORY_CHECK_INTERVAL = 20
PERFORMANCE_SNAPSHOT_INTERVAL = 15

class IntegratedTestRunner:
    
    def __init__(self, test_size: int = DEFAULT_TEST_SIZE, use_finetuned: bool = False, 
                 enable_detailed_monitoring: bool = True):
        """통합 테스트 실행기 초기화"""
        self.test_size = max(MIN_TEST_SIZE, min(test_size, MAX_TEST_SIZE))
        self.use_finetuned = use_finetuned
        self.enable_detailed_monitoring = enable_detailed_monitoring
        self.start_time = time.time()
        
        # 실시간 모니터링
        self.progress_queue = queue.Queue()
        self.performance_snapshots = []
        self.current_question_stats = {}
        
        print(f"통합 추론 테스트 실행기 초기화 중... (대상: {self.test_size}문항)")
        print(f"상세 모니터링: {'활성화' if enable_detailed_monitoring else '비활성화'}")
        
        # 파인튜닝 모델 경로 확인
        if use_finetuned and not os.path.exists("./finetuned_model"):
            print("파인튜닝 모델을 찾을 수 없습니다. 기본 모델을 사용합니다.")
            self.use_finetuned = False
        
        # inference.py의 FinancialAIInference 사용 (통합 추론 기능 포함)
        try:
            print("통합 추론 엔진 초기화 중...")
            self.inference_engine = FinancialAIInference(
                enable_learning=True,
                verbose=False,  # 테스트에서는 간결한 출력
                use_finetuned=self.use_finetuned
            )
            print("통합 추론 엔진 초기화 완료")
        except Exception as e:
            raise RuntimeError(f"통합 추론 엔진 초기화 실패: {e}")
        
        model_type = "파인튜닝된 모델" if self.use_finetuned else "기본 모델"
        reasoning_status = "활성화" if self.inference_engine.reasoning_engine else "비활성화"
        print(f"초기화 완료 - {model_type} 사용, 추론 엔진: {reasoning_status}\n")
    
    def load_test_data(self, test_file: str, submission_file: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """테스트 데이터 로드 및 사전 분석"""
        try:
            if not os.path.exists(test_file):
                print(f"오류: {test_file} 파일을 찾을 수 없습니다")
                return None
            
            if not os.path.exists(submission_file):
                print(f"오류: {submission_file} 파일을 찾을 수 없습니다")
                return None
            
            print("데이터 로드 및 사전 분석 중...")
            test_df = pd.read_csv(test_file, encoding='utf-8')
            submission_df = pd.read_csv(submission_file, encoding='utf-8')
            
            if len(test_df) < self.test_size:
                print(f"경고: 전체 {len(test_df)}문항, 요청 {self.test_size}문항")
                self.test_size = len(test_df)
            
            test_sample = test_df.head(self.test_size).copy()
            submission_sample = submission_df.head(self.test_size).copy()
            
            print(f"테스트 데이터 로드 완료: {len(test_sample)}문항")
            
            # 문제 사전 분석
            if self.enable_detailed_monitoring:
                self._preanalyze_questions(test_sample)
            
            return test_sample, submission_sample
            
        except Exception as e:
            print(f"오류: 데이터 로드 실패 - {e}")
            return None
    
    def _preanalyze_questions(self, test_df: pd.DataFrame) -> None:
        """문제 사전 분석 (복잡도 및 예상 처리시간 계산)"""
        print("문제 사전 분석 수행 중...")
        
        complexity_scores = []
        estimated_times = []
        question_types = {"multiple_choice": 0, "subjective": 0}
        
        for idx, row in test_df.iterrows():
            try:
                question = row['Question']
                
                # 구조 분석 (실제 딥러닝 분석 수행)
                structure = self.inference_engine.data_processor.analyze_question_structure(question)
                
                complexity = structure.get("complexity_score", 0.5)
                complexity_scores.append(complexity)
                
                # 예상 처리시간 계산
                base_time = 8.0 if structure["question_type"] == "multiple_choice" else 15.0
                estimated_time = base_time * (1 + complexity)
                estimated_times.append(estimated_time)
                
                question_types[structure["question_type"]] += 1
                
            except Exception as e:
                print(f"문제 {idx} 분석 오류: {e}")
                complexity_scores.append(0.5)
                estimated_times.append(10.0)
        
        avg_complexity = np.mean(complexity_scores)
        total_estimated_time = sum(estimated_times)
        
        print(f"사전 분석 완료:")
        print(f"  - 객관식: {question_types['multiple_choice']}개")
        print(f"  - 주관식: {question_types['subjective']}개")
        print(f"  - 평균 복잡도: {avg_complexity:.2f}")
        print(f"  - 예상 처리시간: {total_estimated_time/60:.1f}분")
        print(f"  - 문항당 평균: {total_estimated_time/self.test_size:.1f}초")
    
    def run_integrated_test(self, test_file: str = "./test.csv", 
                          submission_file: str = "./sample_submission.csv") -> None:
        """통합 추론 테스트 실행"""
        print("="*60)
        print(f"통합 추론 테스트 시작 ({self.test_size}문항)")
        if self.use_finetuned:
            print("파인튜닝된 모델 사용")
        print("추론 엔진, 학습 시스템, CoT 프롬프트 모두 활성화")
        print("="*60)
        
        # 데이터 로드
        data_result = self.load_test_data(test_file, submission_file)
        if data_result is None:
            return
        
        test_df, submission_df = data_result
        
        print(f"\n통합 딥러닝 추론 시작...")
        print("실제 GPU 추론, CoT 생성, 학습 업데이트 모두 활성화")
        
        # 실시간 모니터링 스레드 시작
        if self.enable_detailed_monitoring:
            monitor_thread = threading.Thread(target=self._performance_monitor, daemon=True)
            monitor_thread.start()
        
        # 답변 생성 - 통합 추론 시스템 사용
        answers = []
        detailed_results = []
        
        try:
            for idx, row in test_df.iterrows():
                question_start_time = time.time()
                question = row['Question']
                question_id = row['ID']
                
                print(f"\n문항 {idx+1}/{self.test_size}: 통합 추론 수행 중...")
                
                # inference.py의 통합 추론 메서드 사용
                answer = self.inference_engine.process_question(question, question_id, idx)
                answers.append(answer)
                
                question_processing_time = time.time() - question_start_time
                
                # 상세 결과 수집
                if self.enable_detailed_monitoring:
                    detailed_result = self._collect_detailed_result(
                        idx, question, answer, question_processing_time
                    )
                    detailed_results.append(detailed_result)
                
                # 진행 상황 업데이트
                if (idx + 1) % PROGRESS_UPDATE_INTERVAL == 0:
                    self._print_progress_update(idx + 1, detailed_results[-PROGRESS_UPDATE_INTERVAL:])
                
                # 성능 스냅샷 수집
                if (idx + 1) % PERFORMANCE_SNAPSHOT_INTERVAL == 0:
                    self._take_performance_snapshot(idx + 1)
            
            # 결과 저장
            submission_df['Answer'] = answers
            
            output_file = f"./integrated_test_result_{self.test_size}.csv"
            submission_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 상세 결과 분석 및 출력
            self._print_comprehensive_results(output_file, detailed_results)
            
        except KeyboardInterrupt:
            print("\n테스트 중단됨")
        except Exception as e:
            print(f"테스트 실행 중 오류: {e}")
            raise
    
    def _collect_detailed_result(self, idx: int, question: str, answer: str, 
                               processing_time: float) -> Dict:
        """상세 결과 수집"""
        stats = self.inference_engine.stats
        
        # 현재 문항의 통계
        detailed_result = {
            "question_idx": idx,
            "question_preview": question[:50] + "..." if len(question) > 50 else question,
            "answer": answer,
            "processing_time": processing_time,
            "model_success": stats.get("model_generation_success", 0) > 0,
            "reasoning_used": stats.get("reasoning_engine_usage", 0) > 0,
            "cot_used": stats.get("cot_prompts_used", 0) > 0,
            "learning_updated": stats.get("learned", 0) > 0,
            "confidence": "high" if stats.get("high_confidence_answers", 0) > 0 else "normal"
        }
        
        # 학습 시스템 정보
        if self.inference_engine.enable_learning:
            learning_stats = self.inference_engine.learning_system.get_learning_statistics()
            detailed_result.update({
                "deep_learning_active": learning_stats.get("deep_learning_active", False),
                "samples_processed": learning_stats.get("samples_processed", 0),
                "gpu_memory_used": learning_stats.get("gpu_memory_used_gb", 0.0)
            })
        
        return detailed_result
    
    def _print_progress_update(self, current: int, recent_results: List[Dict]) -> None:
        """진행 상황 업데이트 출력"""
        if not recent_results:
            return
        
        progress_pct = (current / self.test_size) * 100
        avg_time = np.mean([r["processing_time"] for r in recent_results])
        
        model_success_rate = np.mean([r["model_success"] for r in recent_results]) * 100
        reasoning_rate = np.mean([r["reasoning_used"] for r in recent_results]) * 100
        cot_rate = np.mean([r["cot_used"] for r in recent_results]) * 100
        
        print(f"  진행: {current}/{self.test_size} ({progress_pct:.1f}%)")
        print(f"  최근 {len(recent_results)}문항 평균: {avg_time:.2f}초/문항")
        print(f"  모델성공 {model_success_rate:.0f}%, 추론엔진 {reasoning_rate:.0f}%, CoT {cot_rate:.0f}%")
        
        # 예상 완료 시간
        if current > 5:  # 충분한 샘플 후 예측
            remaining = self.test_size - current
            eta_seconds = remaining * avg_time
            eta_minutes = eta_seconds / 60
            print(f"  예상 완료시간: {eta_minutes:.1f}분 후")
    
    def _performance_monitor(self) -> None:
        """실시간 성능 모니터링 (백그라운드 스레드)"""
        last_check = time.time()
        
        while True:
            time.sleep(5)  # 5초마다 체크
            
            current_time = time.time()
            if current_time - last_check >= MEMORY_CHECK_INTERVAL:
                # GPU 메모리 사용량 체크
                try:
                    import torch
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / (1024**3)
                        if memory_used > 14.0:  # 16GB의 87.5% 초과시 경고
                            print(f"  [경고] GPU 메모리 사용량 높음: {memory_used:.1f}GB")
                except:
                    pass
                
                last_check = current_time
    
    def _take_performance_snapshot(self, current_idx: int) -> None:
        """성능 스냅샷 수집"""
        stats = self.inference_engine.stats
        
        snapshot = {
            "timestamp": time.time(),
            "processed_questions": current_idx,
            "total_time": time.time() - self.start_time,
            "model_success_rate": stats.get("model_generation_success", 0) / max(current_idx, 1),
            "reasoning_usage_rate": stats.get("reasoning_engine_usage", 0) / max(current_idx, 1),
            "cot_usage_rate": stats.get("cot_prompts_used", 0) / max(current_idx, 1),
            "learning_samples": stats.get("learned", 0),
            "avg_processing_time": np.mean(stats.get("processing_times", [1.0]))
        }
        
        self.performance_snapshots.append(snapshot)
    
    def _print_comprehensive_results(self, output_file: str, detailed_results: List[Dict]) -> None:
        """종합적인 결과 분석 및 출력"""
        total_time = time.time() - self.start_time
        
        print(f"\n" + "="*60)
        print("통합 추론 테스트 완료")
        print("="*60)
        
        # 기본 처리 정보
        print(f"총 처리시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"문항당 평균: {total_time/self.test_size:.2f}초")
        
        # 모델 정보
        model_type = "파인튜닝된 모델" if self.use_finetuned else "기본 모델"
        reasoning_status = "활성화" if self.inference_engine.reasoning_engine else "비활성화"
        print(f"사용 모델: {model_type}, 추론 엔진: {reasoning_status}")
        
        # inference.py의 상세 통계 활용
        self._print_integrated_statistics()
        
        # 상세 결과 분석
        if detailed_results:
            self._analyze_detailed_results(detailed_results)
        
        # 성능 스냅샷 분석
        if self.performance_snapshots:
            self._analyze_performance_trends()
        
        print(f"\n결과 파일: {output_file}")
        print("="*60)
    
    def _print_integrated_statistics(self) -> None:
        """통합 통계 출력 (inference.py 통계 활용)"""
        stats = self.inference_engine.stats
        
        print(f"\n🔥 통합 추론 성능:")
        print(f"  모델 생성 성공: {stats['model_generation_success']}/{stats['total']} ({stats['model_generation_success']/max(stats['total'],1)*100:.1f}%)")
        print(f"  추론 엔진 사용: {stats['reasoning_engine_usage']}/{stats['total']} ({stats['reasoning_engine_usage']/max(stats['total'],1)*100:.1f}%)")
        print(f"  CoT 프롬프트 사용: {stats['cot_prompts_used']}/{stats['total']} ({stats['cot_prompts_used']/max(stats['total'],1)*100:.1f}%)")
        print(f"  고신뢰도 답변: {stats['high_confidence_answers']}/{stats['total']} ({stats['high_confidence_answers']/max(stats['total'],1)*100:.1f}%)")
        print(f"  폴백 사용: {stats['fallback_used']}/{stats['total']} ({stats['fallback_used']/max(stats['total'],1)*100:.1f}%)")
        
        # 추론 엔진 상세 통계
        if self.inference_engine.reasoning_engine:
            print(f"\n🧠 추론 엔진 상세:")
            print(f"  추론 성공: {stats['reasoning_successful']}회")
            print(f"  추론 실패: {stats['reasoning_failed']}회")
            print(f"  하이브리드 접근: {stats['hybrid_approach_used']}회")
            print(f"  검증 통과: {stats['verification_passed']}회")
            print(f"  검증 실패: {stats['verification_failed']}회")
            
            if stats['reasoning_time']:
                avg_reasoning_time = np.mean(stats['reasoning_time'])
                print(f"  평균 추론 시간: {avg_reasoning_time:.3f}초")
            
            if stats['reasoning_chain_lengths']:
                avg_chain_length = np.mean(stats['reasoning_chain_lengths'])
                print(f"  평균 추론 체인 길이: {avg_chain_length:.1f}단계")
        
        # 파인튜닝 모델 통계
        if self.use_finetuned:
            finetuned_rate = stats['finetuned_usage'] / max(stats['total'], 1) * 100
            print(f"\n⚡ 파인튜닝 모델 사용률: {finetuned_rate:.1f}%")
        
        # 학습 시스템 통계
        if self.inference_engine.enable_learning:
            print(f"\n📚 딥러닝 학습 시스템:")
            learning_stats = self.inference_engine.learning_system.get_learning_statistics()
            print(f"  학습된 샘플: {stats['learned']}개")
            print(f"  딥러닝 활성화: {learning_stats['deep_learning_active']}")
            print(f"  처리된 샘플: {learning_stats['samples_processed']}개")
            print(f"  가중치 업데이트: {learning_stats['weights_updated']}회")
            print(f"  GPU 메모리 사용: {learning_stats['gpu_memory_used_gb']:.2f}GB")
            print(f"  총 학습 시간: {learning_stats['total_training_time']:.1f}초")
            if learning_stats['average_loss'] > 0:
                print(f"  평균 손실: {learning_stats['average_loss']:.4f}")
            print(f"  현재 정확도: {self.inference_engine.learning_system.get_current_accuracy():.2%}")
        
        # 한국어 품질 통계
        if stats['quality_scores']:
            avg_quality = np.mean(stats['quality_scores'])
            quality_level = "우수" if avg_quality > 0.8 else "양호" if avg_quality > 0.65 else "개선 필요"
            print(f"\n🇰🇷 한국어 품질: {avg_quality:.2f} ({quality_level})")
        
        # 답변 분포
        distribution = stats['answer_distribution']
        total_mc = sum(distribution.values())
        if total_mc > 0:
            print(f"\n📊 객관식 답변 분포:")
            for ans in sorted(distribution.keys()):
                count = distribution[ans]
                if count > 0:
                    pct = count / total_mc * 100
                    print(f"  {ans}번: {count}개 ({pct:.1f}%)")
            
            unique_answers = len([k for k, v in distribution.items() if v > 0])
            diversity = "우수" if unique_answers >= 4 else "양호" if unique_answers >= 3 else "개선 필요"
            print(f"  답변 다양성: {diversity} ({unique_answers}/5개 번호 사용)")
    
    def _analyze_detailed_results(self, detailed_results: List[Dict]) -> None:
        """상세 결과 분석"""
        if not detailed_results:
            return
        
        print(f"\n📈 상세 성능 분석:")
        
        # 처리시간 분석
        processing_times = [r["processing_time"] for r in detailed_results]
        print(f"  처리시간 - 최소: {min(processing_times):.2f}초, 최대: {max(processing_times):.2f}초")
        print(f"  처리시간 - 평균: {np.mean(processing_times):.2f}초, 중앙값: {np.median(processing_times):.2f}초")
        
        # 성공률 분석
        model_success_rate = np.mean([r["model_success"] for r in detailed_results]) * 100
        reasoning_rate = np.mean([r["reasoning_used"] for r in detailed_results]) * 100
        cot_rate = np.mean([r["cot_used"] for r in detailed_results]) * 100
        learning_rate = np.mean([r["learning_updated"] for r in detailed_results]) * 100
        
        print(f"  성공률 - 모델: {model_success_rate:.1f}%, 추론: {reasoning_rate:.1f}%, CoT: {cot_rate:.1f}%, 학습: {learning_rate:.1f}%")
        
        # 신뢰도 분석
        high_conf_rate = np.mean([r["confidence"] == "high" for r in detailed_results]) * 100
        print(f"  고신뢰도 답변 비율: {high_conf_rate:.1f}%")
        
        # 딥러닝 학습 분석
        if any("deep_learning_active" in r for r in detailed_results):
            dl_active_rate = np.mean([r.get("deep_learning_active", False) for r in detailed_results]) * 100
            avg_samples = np.mean([r.get("samples_processed", 0) for r in detailed_results])
            avg_gpu_memory = np.mean([r.get("gpu_memory_used", 0.0) for r in detailed_results])
            
            print(f"  딥러닝 활성화율: {dl_active_rate:.1f}%")
            print(f"  평균 처리 샘플: {avg_samples:.1f}개")
            print(f"  평균 GPU 메모리: {avg_gpu_memory:.2f}GB")
    
    def _analyze_performance_trends(self) -> None:
        """성능 트렌드 분석"""
        if len(self.performance_snapshots) < 2:
            return
        
        print(f"\n📊 성능 트렌드 분석:")
        
        # 처리속도 트렌드
        early_snapshot = self.performance_snapshots[0]
        late_snapshot = self.performance_snapshots[-1]
        
        speed_change = late_snapshot["avg_processing_time"] - early_snapshot["avg_processing_time"]
        speed_trend = "향상" if speed_change < 0 else "저하" if speed_change > 0 else "안정"
        
        print(f"  처리속도 트렌드: {speed_trend} ({speed_change:+.2f}초)")
        
        # 성공률 트렌드
        success_change = late_snapshot["model_success_rate"] - early_snapshot["model_success_rate"]
        success_trend = "향상" if success_change > 0 else "저하" if success_change < 0 else "안정"
        
        print(f"  모델 성공률 트렌드: {success_trend} ({success_change:+.1%})")
        
        # 학습 진행상황
        learning_progress = late_snapshot["learning_samples"] - early_snapshot["learning_samples"]
        print(f"  학습 진행: +{learning_progress}개 샘플")
    
    def get_integration_test_summary(self) -> Dict:
        """통합 테스트 요약"""
        stats = self.inference_engine.stats
        
        if stats["total"] == 0:
            return {"error": "아직 테스트가 실행되지 않았습니다"}
        
        summary = {
            "총_문항": stats["total"],
            "모델_성공률": f"{stats['model_generation_success']/stats['total']*100:.1f}%",
            "추론_엔진_사용률": f"{stats['reasoning_engine_usage']/stats['total']*100:.1f}%",
            "CoT_사용률": f"{stats['cot_prompts_used']/stats['total']*100:.1f}%",
            "학습_샘플": stats['learned'],
            "폴백_사용률": f"{stats['fallback_used']/stats['total']*100:.1f}%",
            "평균_처리시간": f"{np.mean(stats['processing_times']):.2f}초" if stats['processing_times'] else "N/A"
        }
        
        if self.use_finetuned:
            summary["파인튜닝_사용률"] = f"{stats['finetuned_usage']/stats['total']*100:.1f}%"
        
        if stats["quality_scores"]:
            summary["한국어_품질"] = f"{np.mean(stats['quality_scores']):.2f}"
        
        if self.inference_engine.enable_learning:
            learning_stats = self.inference_engine.learning_system.get_learning_statistics()
            summary["딥러닝_활성화"] = learning_stats['deep_learning_active']
            summary["GPU_메모리_사용"] = f"{learning_stats['gpu_memory_used_gb']:.1f}GB"
            summary["학습_정확도"] = f"{self.inference_engine.learning_system.get_current_accuracy():.1%}"
        
        return summary
    
    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            print("\n시스템 정리 중...")
            
            # inference.py의 cleanup 메서드 사용
            if hasattr(self, 'inference_engine'):
                self.inference_engine.cleanup()
            
            # 성능 데이터 정리
            self.performance_snapshots.clear()
            
            print("정리 완료")
            
        except Exception as e:
            print(f"정리 중 오류: {e}")


def main():
    """메인 함수"""
    test_size = DEFAULT_TEST_SIZE
    use_finetuned = False
    enable_monitoring = True
    
    # 명령행 인수 처리
    if len(sys.argv) > 1:
        try:
            test_size = int(sys.argv[1])
            test_size = max(MIN_TEST_SIZE, min(test_size, MAX_TEST_SIZE))
        except ValueError:
            print("잘못된 문항 수, 기본값 50 사용")
            test_size = DEFAULT_TEST_SIZE
    
    if len(sys.argv) > 2:
        if sys.argv[2].lower() in ['true', '1', 'yes', 'finetuned']:
            use_finetuned = True
    
    if len(sys.argv) > 3:
        if sys.argv[3].lower() in ['false', '0', 'no', 'simple']:
            enable_monitoring = False
    
    # 파인튜닝 모델 자동 감지
    if os.path.exists("./finetuned_model") and not use_finetuned:
        try:
            response = input("파인튜닝된 모델이 발견되었습니다. 사용하시겠습니까? (y/n): ")
            if response.lower() in ['y', 'yes']:
                use_finetuned = True
        except (EOFError, KeyboardInterrupt):
            print("\n기본 모델 사용")
    
    print(f"통합 추론 테스트 실행기 시작 (Python {sys.version.split()[0]})")
    print(f"GPU 기반 딥러닝 추론 및 학습 시스템 활성화")
    
    runner = None
    try:
        runner = IntegratedTestRunner(
            test_size=test_size, 
            use_finetuned=use_finetuned,
            enable_detailed_monitoring=enable_monitoring
        )
        runner.run_integrated_test()
        
        # 통합 성능 요약 출력
        summary = runner.get_integration_test_summary()
        print(f"\n🎯 통합 테스트 성능 요약:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # 성공 여부 판단
        if summary.get("모델_성공률", "0%") == "0.0%":
            print(f"\n⚠️  경고: 모델 생성 성공률이 0%입니다. 실제 GPU 추론이 작동하지 않을 수 있습니다.")
        elif float(summary.get("모델_성공률", "0%").rstrip("%")) > 70:
            print(f"\n✅ 성공: 통합 추론 시스템이 정상 작동하고 있습니다.")
        else:
            print(f"\n⚠️  주의: 모델 성공률이 낮습니다. 시스템 점검이 필요할 수 있습니다.")
        
    except KeyboardInterrupt:
        print("\n테스트 중단")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if runner:
            runner.cleanup()


if __name__ == "__main__":
    main()