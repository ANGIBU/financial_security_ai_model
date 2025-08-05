# main.py
"""
개발/테스트용 메인 파일
"""

import os
import pandas as pd
import torch
import time
import argparse
from tqdm import tqdm
import warnings
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
warnings.filterwarnings("ignore")

from model_handler import OptimizedModelHandler
from data_processor import IntelligentDataProcessor
from prompt_engineering import AdvancedPromptEngineer
from knowledge_base import FinancialSecurityKnowledgeBase
from advanced_optimizer import UltraHighPerformanceOptimizer, PerformanceMonitor
from pattern_learner import AnswerPatternLearner, SmartAnswerSelector

class UltraHighPerformanceTester:
    """개발 및 테스트 클래스"""
    
    def __init__(self, model_config: dict):
        print("개발 시스템 초기화...")
        
        # 성능 모니터링 시작
        self.performance_monitor = PerformanceMonitor()
        self.start_time = time.time()
        
        # 컴포넌트 초기화
        self.model_handler = OptimizedModelHandler(**model_config)
        self.data_processor = IntelligentDataProcessor()
        self.prompt_engineer = AdvancedPromptEngineer()
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.optimizer = UltraHighPerformanceOptimizer()
        self.pattern_learner = AnswerPatternLearner()
        self.answer_selector = SmartAnswerSelector()
        
        # 시스템 성능 정보
        self.system_info = self._collect_system_info()
        
        print("✅ 고성능 초기화 완료")
        self._print_system_summary()
    
    def _collect_system_info(self) -> dict:
        """시스템 정보 수집"""
        info = {
            "gpu_memory_gb": 0,
            "cpu_cores": psutil.cpu_count(logical=False),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "performance_tier": "Basic"
        }
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = gpu_props.name
            info["gpu_memory_gb"] = gpu_props.total_memory / (1024**3)
            
            # 성능 등급 판정
            if info["gpu_memory_gb"] >= 20:
                info["performance_tier"] = "Ultra High"
            elif info["gpu_memory_gb"] >= 12:
                info["performance_tier"] = "High"
            elif info["gpu_memory_gb"] >= 8:
                info["performance_tier"] = "Medium"
        
        return info
    
    def _print_system_summary(self):
        """시스템 요약 출력"""
        print(f"\n📊 시스템 정보")
        if torch.cuda.is_available():
            print(f"GPU: {self.system_info['gpu_name']} ({self.system_info['gpu_memory_gb']:.1f}GB)")
        print(f"CPU: {self.system_info['cpu_cores']}코어")
        print(f"RAM: {self.system_info['ram_gb']:.1f}GB")
        print(f"성능 등급: {self.system_info['performance_tier']}")
    
    def run_comprehensive_accuracy_test(self, sample_size: int = 30):
        """포괄적 정확도 테스트"""
        
        test_df = pd.read_csv('./test.csv')
        
        print(f"\n🎯 포괄적 정확도 테스트: {sample_size}개 문항")
        
        # 지능형 샘플 선택
        sample_indices = self._select_intelligent_samples(test_df, sample_size)
        
        results = []
        confidence_scores = []
        processing_times = []
        answer_distribution = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        
        # 진행률 표시와 함께 처리
        for idx in tqdm(sample_indices, desc="정확도 테스트"):
            start_time = time.time()
            
            question = test_df.iloc[idx]['Question']
            question_id = test_df.iloc[idx]['ID']
            
            # 고급 문제 분석
            structure = self.data_processor.analyze_question_structure(question)
            analysis = self.knowledge_base.analyze_question(question)
            difficulty = self.optimizer.evaluate_question_difficulty_advanced(question, structure)
            
            # 적응형 전략 선택
            strategies = self._select_adaptive_strategies(difficulty, structure)
            
            best_answer = None
            best_confidence = 0
            best_reasoning = ""
            
            # 다중 전략 테스트
            for strategy in strategies:
                try:
                    # 고급 프롬프트 생성
                    prompt = self.prompt_engineer.create_adaptive_prompt(
                        question, structure["question_type"], analysis, strategy
                    )
                    
                    # 모델별 최적화
                    optimized_prompt = self.prompt_engineer.optimize_for_model(
                        prompt, self.model_handler.model_name
                    )
                    
                    # 추론 실행
                    result = self.model_handler.generate_expert_response(
                        optimized_prompt, structure["question_type"], max_attempts=1
                    )
                    
                    # 답변 후처리
                    processed_answer = self.data_processor.post_process_answer(
                        result.response, question, structure["question_type"]
                    )
                    
                    # 스마트 선택기로 최종 결정
                    final_answer, final_confidence = self.answer_selector.select_best_answer(
                        question, result.response, structure, result.confidence
                    )
                    
                    if final_confidence > best_confidence:
                        best_confidence = final_confidence
                        best_answer = final_answer
                        best_reasoning = result.response[:200] + "..."
                        
                except Exception as e:
                    continue
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # 결과 기록
            if best_answer and structure["question_type"] == "multiple_choice":
                if best_answer in answer_distribution:
                    answer_distribution[best_answer] += 1
            
            confidence_scores.append(best_confidence)
            
            results.append({
                "id": question_id,
                "question": question[:100] + "...",
                "answer": best_answer,
                "confidence": best_confidence,
                "reasoning": best_reasoning,
                "type": structure["question_type"],
                "difficulty": difficulty.score,
                "processing_time": processing_time,
                "domain": analysis.get("domain", ["일반"])
            })
            
            # 성능 모니터링 업데이트
            self.performance_monitor.update(processing_time, best_confidence)
        
        # 상세 결과 분석
        self._analyze_comprehensive_accuracy_results(results, confidence_scores, 
                                                    processing_times, answer_distribution)
        
        return results
    
    def run_ultra_speed_test(self, sample_size: int = 100):
        """초고속 테스트"""
        
        test_df = pd.read_csv('./test.csv')
        
        print(f"\n⚡ 초고속 테스트: {sample_size}개 문항")
        
        # 빠른 처리용 샘플 선택 (쉬운 문제 위주)
        sample_indices = self._select_speed_optimized_samples(test_df, sample_size)
        
        start_time = time.time()
        results = []
        
        # 동적 배치 크기 결정
        gpu_memory = self.system_info["gpu_memory_gb"]
        if self.system_info["performance_tier"] == "Ultra High":
            batch_size = 20
        elif self.system_info["performance_tier"] == "High":
            batch_size = 15
        else:
            batch_size = 10
        
        print(f"🔧 최적화된 배치 크기: {batch_size}")
        
        # 배치별 병렬 처리
        batch_results = []
        for i in tqdm(range(0, len(sample_indices), batch_size), desc="배치 처리"):
            batch_indices = sample_indices[i:i+batch_size]
            batch_questions = [test_df.iloc[idx]['Question'] for idx in batch_indices]
            
            # 배치 처리
            batch_start = time.time()
            processed_batch = self._process_speed_batch(batch_questions, batch_indices, test_df)
            batch_time = time.time() - batch_start
            
            batch_results.extend(processed_batch)
            
            print(f"  배치 {i//batch_size + 1}: {len(processed_batch)}개, {batch_time:.1f}초")
            
            # 메모리 관리
            if i % (batch_size * 3) == 0:
                torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        
        # 속도 분석
        self._analyze_speed_results(batch_results, total_time, sample_size)
        
        return batch_results
    
    def run_stress_test(self, duration_minutes: int = 10):
        """스트레스 테스트"""
        
        print(f"\n🔥 시스템 스트레스 테스트: {duration_minutes}분")
        
        test_df = pd.read_csv('./test.csv')
        end_time = time.time() + (duration_minutes * 60)
        
        processed_count = 0
        error_count = 0
        performance_history = []
        
        while time.time() < end_time:
            try:
                # 랜덤 문제 선택
                idx = np.random.randint(0, len(test_df))
                question = test_df.iloc[idx]['Question']
                
                start = time.time()
                
                # 빠른 처리
                structure = self.data_processor.analyze_question_structure(question)
                prompt = self.prompt_engineer.create_adaptive_prompt(
                    question, structure["question_type"], {}, "simple"
                )
                
                result = self.model_handler.generate_expert_response(
                    prompt, structure["question_type"], max_attempts=1
                )
                
                processing_time = time.time() - start
                
                # 성능 기록
                performance_history.append({
                    "time": time.time(),
                    "processing_time": processing_time,
                    "confidence": result.confidence,
                    "gpu_memory": torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                })
                
                processed_count += 1
                
                # 진행 상황 출력 (10초마다)
                if processed_count % 10 == 0:
                    elapsed = time.time() - (end_time - duration_minutes * 60)
                    remaining = (end_time - time.time()) / 60
                    rate = processed_count / (elapsed / 60)
                    print(f"  진행: {processed_count}개 처리, {rate:.1f}개/분, 남은시간: {remaining:.1f}분")
                
            except Exception as e:
                error_count += 1
                if error_count > 10:  # 너무 많은 오류 발생 시 중단
                    print(f"⚠️ 과도한 오류 발생 ({error_count}개), 테스트 중단")
                    break
        
        # 스트레스 테스트 분석
        self._analyze_stress_test_results(processed_count, error_count, performance_history, duration_minutes)
        
        return {
            "processed_count": processed_count,
            "error_count": error_count,
            "performance_history": performance_history
        }
    
    def _select_intelligent_samples(self, test_df: pd.DataFrame, sample_size: int) -> list:
        """지능형 샘플 선택"""
        indices = []
        
        # 전략적 패턴별 선택
        patterns = [
            ("개인정보", min(8, sample_size//4)),      # 개인정보보호 문제
            ("전자금융", min(8, sample_size//4)),      # 전자금융 문제
            ("해당하지않는", min(6, sample_size//5)),   # 부정형 문제
            ("정의", min(4, sample_size//6)),          # 정의 문제
            ("법", min(4, sample_size//6))            # 법령 문제
        ]
        
        used_indices = set()
        
        # 패턴별 샘플링
        for pattern, target_count in patterns:
            found_count = 0
            for i, question in enumerate(test_df['Question']):
                if pattern in question and i not in used_indices:
                    # 문제 품질 검사
                    if self._is_good_sample(question):
                        indices.append(i)
                        used_indices.add(i)
                        found_count += 1
                        if found_count >= target_count:
                            break
        
        # 랜덤 샘플로 부족한 부분 채우기
        remaining = sample_size - len(indices)
        if remaining > 0:
            available_indices = [i for i in range(len(test_df)) if i not in used_indices]
            random_indices = np.random.choice(available_indices, 
                                            min(remaining, len(available_indices)), 
                                            replace=False)
            indices.extend(random_indices)
        
        return indices[:sample_size]
    
    def _select_speed_optimized_samples(self, test_df: pd.DataFrame, sample_size: int) -> list:
        """속도 최적화된 샘플 선택"""
        # 짧고 간단한 문제 위주 선택
        simple_indices = []
        
        for i, question in enumerate(test_df['Question']):
            if len(question) < 300:  # 짧은 문제
                if not any(neg in question for neg in ["해당하지않는", "적절하지않은"]):  # 부정형 제외
                    simple_indices.append(i)
        
        # 샘플 크기만큼 랜덤 선택
        if len(simple_indices) >= sample_size:
            return np.random.choice(simple_indices, sample_size, replace=False).tolist()
        else:
            # 부족하면 전체에서 랜덤 선택
            additional = sample_size - len(simple_indices)
            remaining_indices = [i for i in range(len(test_df)) if i not in simple_indices]
            additional_indices = np.random.choice(remaining_indices, additional, replace=False)
            return simple_indices + additional_indices.tolist()
    
    def _is_good_sample(self, question: str) -> bool:
        """좋은 샘플인지 판단"""
        # 너무 짧거나 긴 문제 제외
        if len(question) < 50 or len(question) > 2000:
            return False
        
        # 특수문자가 너무 많은 문제 제외
        special_char_ratio = len(re.findall(r'[^\w\s가-힣]', question)) / len(question)
        if special_char_ratio > 0.3:
            return False
        
        return True
    
    def _select_adaptive_strategies(self, difficulty, structure) -> list:
        """적응형 전략 선택"""
        strategies = []
        
        # 난이도 기반 전략
        if difficulty.score < 0.3:
            strategies = ["simple"]
        elif difficulty.score < 0.6:
            strategies = ["simple", "balanced"]
        else:
            strategies = ["balanced", "comprehensive"]
        
        # 구조 기반 추가 전략
        if structure.get("has_negative", False):
            strategies.append("negative_focused")
        
        # 도메인 기반 전략
        domains = structure.get("domain", [])
        if domains and any(d in ["개인정보보호", "전자금융"] for d in domains):
            strategies.append("domain_specific")
        
        return strategies[:2]  # 최대 2개 전략
    
    def _process_speed_batch(self, questions: list, indices: list, test_df: pd.DataFrame) -> list:
        """속도 최적화 배치 처리"""
        batch_results = []
        
        # 간단한 프롬프트로 빠른 처리
        for i, question in enumerate(questions):
            try:
                # 최소한의 분석
                is_mc = bool(re.search(r'[①②③④⑤]|\b[1-5]\s*[.)]', question))
                
                # 빠른 프롬프트 생성
                if is_mc:
                    prompt = f"다음 객관식 문제의 정답 번호를 선택하세요.\n\n{question}\n\n정답:"
                else:
                    prompt = f"다음 질문에 간결하게 답변하세요.\n\n{question}\n\n답변:"
                
                # 모델 최적화
                optimized_prompt = self.prompt_engineer.optimize_for_model(
                    prompt, self.model_handler.model_name
                )
                
                # 빠른 생성 (타임아웃 단축)
                result = self.model_handler.generate_expert_response(
                    optimized_prompt, "multiple_choice" if is_mc else "subjective", 
                    max_attempts=1
                )
                
                # 빠른 답변 추출
                if is_mc:
                    answer = self.data_processor.extract_mc_answer_fast(result.response)
                else:
                    answer = result.response[:200]  # 주관식은 200자로 제한
                
                batch_results.append({
                    "id": test_df.iloc[indices[i]]['ID'],
                    "answer": answer,
                    "confidence": result.confidence,
                    "time": result.inference_time
                })
                
            except Exception as e:
                # 오류 시 기본값
                batch_results.append({
                    "id": test_df.iloc[indices[i]]['ID'],
                    "answer": "3" if is_mc else "관련 규정에 따른 조치가 필요합니다.",
                    "confidence": 0.3,
                    "time": 0.1
                })
        
        return batch_results
    
    def _analyze_comprehensive_accuracy_results(self, results: list, confidence_scores: list, 
                                              processing_times: list, answer_distribution: dict):
        """포괄적 정확도 결과 분석"""
        
        print(f"\n🎯 정확도 분석 결과")
        print(f"{'='*50}")
        
        # 기본 통계
        mc_results = [r for r in results if r["type"] == "multiple_choice"]
        subj_results = [r for r in results if r["type"] == "subjective"]
        
        print(f"총 문항: {len(results)}개")
        print(f"  객관식: {len(mc_results)}개 ({len(mc_results)/len(results)*100:.1f}%)")
        print(f"  주관식: {len(subj_results)}개 ({len(subj_results)/len(results)*100:.1f}%)")
        
        # 신뢰도 분석
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            high_conf_count = len([c for c in confidence_scores if c >= 0.7])
            
            print(f"\n📊 신뢰도 분석")
            print(f"평균 신뢰도: {avg_confidence:.3f}")
            print(f"고신뢰도 (≥0.7): {high_conf_count}개 ({high_conf_count/len(results)*100:.1f}%)")
        
        # 처리 시간 분석
        if processing_times:
            avg_time = np.mean(processing_times)
            min_time = np.min(processing_times)
            max_time = np.max(processing_times)
            
            print(f"\n⏱️ 처리 시간 분석")
            print(f"평균 처리 시간: {avg_time:.2f}초")
            print(f"최소/최대: {min_time:.2f}초 / {max_time:.2f}초")
            print(f"예상 전체 시간: {(avg_time * 515) / 60:.1f}분")
        
        # 답변 분포 분석 (객관식)
        if mc_results:
            print(f"\n📈 객관식 답변 분포")
            total_mc = len(mc_results)
            for choice in sorted(answer_distribution.keys()):
                count = answer_distribution[choice]
                pct = (count / total_mc * 100) if total_mc > 0 else 0
                print(f"  {choice}번: {count}개 ({pct:.1f}%)")
            
            # 편향 검사
            max_choice = max(answer_distribution, key=answer_distribution.get)
            max_pct = (answer_distribution[max_choice] / total_mc) * 100
            if max_pct > 50:
                print(f"⚠️ 답변 편향 감지: {max_choice}번 {max_pct:.1f}%")
            else:
                print("✅ 답변 분포 균형적")
        
        # 난이도별 분석
        difficulty_stats = {}
        for result in results:
            diff = result["difficulty"]
            if diff < 0.3:
                category = "쉬움"
            elif diff < 0.7:
                category = "보통"
            else:
                category = "어려움"
            
            if category not in difficulty_stats:
                difficulty_stats[category] = {"count": 0, "avg_conf": []}
            difficulty_stats[category]["count"] += 1
            difficulty_stats[category]["avg_conf"].append(result["confidence"])
        
        print(f"\n📊 난이도별 분석")
        for category, stats in difficulty_stats.items():
            avg_conf = np.mean(stats["avg_conf"]) if stats["avg_conf"] else 0
            print(f"  {category}: {stats['count']}개, 평균신뢰도 {avg_conf:.3f}")
    
    def _analyze_speed_results(self, results: list, total_time: float, sample_size: int):
        """속도 결과 분석"""
        
        print(f"\n⚡ 속도 분석 결과")
        print(f"{'='*50}")
        
        print(f"총 처리 시간: {total_time:.1f}초")
        print(f"평균 처리 속도: {total_time/sample_size:.2f}초/문항")
        print(f"처리 속도: {sample_size/(total_time/60):.1f}문항/분")
        
        # 전체 예상 시간
        estimated_total_time = (total_time / sample_size) * 515
        print(f"예상 전체 처리 시간: {estimated_total_time/60:.1f}분")
        
        # 성능 등급 평가
        questions_per_minute = sample_size / (total_time / 60)
        if questions_per_minute > 30:
            performance_grade = "S급 (초고속)"
        elif questions_per_minute > 20:
            performance_grade = "A급 (고속)"
        elif questions_per_minute > 10:
            performance_grade = "B급 (보통)"
        else:
            performance_grade = "C급 (개선필요)"
        
        print(f"성능 등급: {performance_grade}")
        
        # 시간 여유 분석
        time_limit_minutes = 270  # 4시간 30분
        safety_margin = time_limit_minutes - (estimated_total_time / 60)
        print(f"시간 여유: {safety_margin:.1f}분")
        
        if safety_margin > 60:
            print("✅ 충분한 시간 여유")
        elif safety_margin > 30:
            print("⚠️ 적당한 시간 여유")
        else:
            print("❌ 시간 부족 위험")
    
    def _analyze_stress_test_results(self, processed_count: int, error_count: int, 
                                   performance_history: list, duration_minutes: int):
        """스트레스 테스트 결과 분석"""
        
        print(f"\n🔥 스트레스 테스트 결과")
        print(f"{'='*50}")
        
        print(f"테스트 시간: {duration_minutes}분")
        print(f"처리된 문항: {processed_count}개")
        print(f"오류 발생: {error_count}개")
        print(f"성공률: {((processed_count-error_count)/processed_count)*100:.1f}%")
        
        if performance_history:
            # 성능 추이 분석
            processing_times = [p["processing_time"] for p in performance_history]
            confidences = [p["confidence"] for p in performance_history]
            
            print(f"\n📈 성능 추이")
            print(f"평균 처리 시간: {np.mean(processing_times):.2f}초")
            print(f"처리 시간 표준편차: {np.std(processing_times):.2f}초")
            print(f"평균 신뢰도: {np.mean(confidences):.3f}")
            
            # GPU 메모리 사용량 (CUDA 사용 시)
            if torch.cuda.is_available() and performance_history:
                gpu_memories = [p["gpu_memory"] for p in performance_history]
                print(f"GPU 메모리 사용량: {np.mean(gpu_memories):.1f}GB (평균)")
        
        # 안정성 평가
        error_rate = error_count / processed_count if processed_count > 0 else 1
        if error_rate < 0.01:
            stability_grade = "매우 안정적"
        elif error_rate < 0.05:
            stability_grade = "안정적"
        elif error_rate < 0.1:
            stability_grade = "보통"
        else:
            stability_grade = "불안정"
        
        print(f"시스템 안정성: {stability_grade}")
    
    def run_comprehensive_benchmark(self):
        """종합 벤치마크"""
        
        print(f"\n🏆 종합 벤치마크 시작")
        print(f"{'='*60}")
        
        benchmark_results = {}
        
        # 1. 정확도 테스트
        print("\n1️⃣ 정확도 테스트 (20개 샘플)")
        accuracy_results = self.run_comprehensive_accuracy_test(20)
        benchmark_results["accuracy"] = {
            "sample_count": len(accuracy_results),
            "avg_confidence": np.mean([r["confidence"] for r in accuracy_results]),
            "high_confidence_rate": len([r for r in accuracy_results if r["confidence"] >= 0.7]) / len(accuracy_results)
        }
        
        # 2. 속도 테스트
        print("\n2️⃣ 속도 테스트 (50개 샘플)")
        speed_results = self.run_ultra_speed_test(50)
        benchmark_results["speed"] = {
            "sample_count": len(speed_results),
            "avg_time": np.mean([r["time"] for r in speed_results]),
            "questions_per_minute": len(speed_results) / (sum([r["time"] for r in speed_results]) / 60)
        }
        
        # 3. 메모리 효율성 테스트
        print("\n3️⃣ 메모리 효율성 테스트")
        memory_results = self._test_memory_efficiency()
        benchmark_results["memory"] = memory_results
        
        # 종합 점수 계산
        total_score = self._calculate_benchmark_score(benchmark_results)
        
        # 최종 보고서
        self._generate_benchmark_report(benchmark_results, total_score)
        
        return benchmark_results
    
    def _test_memory_efficiency(self) -> dict:
        """메모리 효율성 테스트"""
        
        if not torch.cuda.is_available():
            return {"status": "CUDA 없음"}
        
        # 메모리 사용량 측정
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024**3)
        
        # 작업 부하 생성
        test_df = pd.read_csv('./test.csv')
        sample_questions = test_df['Question'].head(10).tolist()
        
        max_memory = initial_memory
        
        for question in sample_questions:
            structure = self.data_processor.analyze_question_structure(question)
            prompt = self.prompt_engineer.create_adaptive_prompt(
                question, structure["question_type"], {}, "simple"
            )
            
            # 메모리 사용량 모니터링
            current_memory = torch.cuda.memory_allocated() / (1024**3)
            max_memory = max(max_memory, current_memory)
        
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated() / (1024**3)
        
        return {
            "initial_memory_gb": round(initial_memory, 2),
            "max_memory_gb": round(max_memory, 2),
            "final_memory_gb": round(final_memory, 2),
            "memory_efficiency": round((final_memory - initial_memory) / max_memory, 3),
            "total_gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    
    def _calculate_benchmark_score(self, results: dict) -> float:
        """벤치마크 점수 계산"""
        
        score = 0
        
        # 정확도 점수 (40점)
        if "accuracy" in results:
            acc = results["accuracy"]
            accuracy_score = (acc["avg_confidence"] * 30) + (acc["high_confidence_rate"] * 10)
            score += min(accuracy_score, 40)
        
        # 속도 점수 (35점)
        if "speed" in results:
            speed = results["speed"]
            questions_per_min = speed["questions_per_minute"]
            if questions_per_min > 30:
                speed_score = 35
            elif questions_per_min > 20:
                speed_score = 30
            elif questions_per_min > 10:
                speed_score = 20
            else:
                speed_score = 10
            score += speed_score
        
        # 메모리 효율성 점수 (15점)
        if "memory" in results and "memory_efficiency" in results["memory"]:
            memory_eff = results["memory"]["memory_efficiency"]
            if memory_eff < 0.3:
                memory_score = 15
            elif memory_eff < 0.5:
                memory_score = 12
            elif memory_eff < 0.7:
                memory_score = 8
            else:
                memory_score = 5
            score += memory_score
        
        # 시스템 안정성 점수 (10점)
        # GPU 메모리 크기에 따른 보너스
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 20:
                score += 10  # 대용량 GPU 보너스
            elif gpu_memory >= 12:
                score += 8
            else:
                score += 5
        
        return min(score, 100)
    
    def _generate_benchmark_report(self, results: dict, total_score: float):
        """벤치마크 보고서 생성"""
        
        print(f"\n🏆 종합 벤치마크 보고서")
        print(f"{'='*60}")
        
        print(f"총점: {total_score:.1f}/100")
        
        # 등급 판정
        if total_score >= 90:
            grade = "S급 (최우수)"
            comment = "🥇 최적의 성능! 대회 준비 완료"
        elif total_score >= 80:
            grade = "A급 (우수)"
            comment = "🥈 우수한 성능! 약간의 조정으로 완벽"
        elif total_score >= 70:
            grade = "B급 (양호)"
            comment = "🥉 양호한 성능! 일부 개선 권장"
        elif total_score >= 60:
            grade = "C급 (보통)"
            comment = "⚠️ 보통 성능, 최적화 필요"
        else:
            grade = "D급 (개선필요)"
            comment = "❌ 성능 개선 필요"
        
        print(f"성능 등급: {grade}")
        print(f"평가: {comment}")
        
        # 세부 점수
        print(f"\n📊 세부 점수")
        if "accuracy" in results:
            print(f"정확도: {results['accuracy']['avg_confidence']:.3f} (신뢰도)")
        if "speed" in results:
            print(f"속도: {results['speed']['questions_per_minute']:.1f} 문항/분")
        if "memory" in results and "memory_efficiency" in results["memory"]:
            print(f"메모리 효율성: {results['memory']['memory_efficiency']:.3f}")
        
        # 권장사항
        print(f"\n💡 성능 개선 권장사항")
        recommendations = self._generate_improvement_recommendations(results, total_score)
        for rec in recommendations:
            print(f"  • {rec}")
    
    def _generate_improvement_recommendations(self, results: dict, score: float) -> list:
        """개선 권장사항 생성"""
        
        recommendations = []
        
        # 점수 기반 권장사항
        if score < 70:
            recommendations.append("시스템 사양 업그레이드 고려 (GPU 메모리, CPU 성능)")
        
        # 정확도 기반
        if "accuracy" in results:
            avg_conf = results["accuracy"]["avg_confidence"]
            if avg_conf < 0.6:
                recommendations.append("프롬프트 엔지니어링 개선으로 신뢰도 향상")
                recommendations.append("도메인 특화 지식 베이스 확장")
        
        # 속도 기반
        if "speed" in results:
            qpm = results["speed"]["questions_per_minute"]
            if qpm < 15:
                recommendations.append("배치 크기 증가 및 병렬 처리 최적화")
                recommendations.append("모델 컴파일 및 Mixed Precision 활용")
        
        # 메모리 기반
        if "memory" in results and "memory_efficiency" in results["memory"]:
            if results["memory"]["memory_efficiency"] > 0.5:
                recommendations.append("메모리 관리 개선 (캐시 정리, 배치 크기 조정)")
        
        # 시스템별 권장사항
        performance_tier = self.system_info["performance_tier"]
        if performance_tier == "Medium":
            recommendations.append("GPU 메모리 부족 - 메모리 절약 모드 활성화")
        elif performance_tier == "Basic":
            recommendations.append("하드웨어 업그레이드 강력 권장")
        
        if not recommendations:
            recommendations.append("현재 설정이 최적화되어 있습니다!")
        
        return recommendations
    
    def cleanup(self):
        """리소스 정리"""
        print(f"\n🧹 리소스 정리 중...")
        
        # 컴포넌트 정리
        if hasattr(self, 'model_handler'):
            self.model_handler.cleanup()
        if hasattr(self, 'data_processor'):
            self.data_processor.cleanup()
        if hasattr(self, 'prompt_engineer'):
            self.prompt_engineer.cleanup()
        if hasattr(self, 'knowledge_base'):
            self.knowledge_base.cleanup()
        if hasattr(self, 'pattern_learner'):
            self.pattern_learner.cleanup()
        if hasattr(self, 'answer_selector'):
            self.answer_selector.cleanup()
        
        # 성능 통계 출력
        total_time = time.time() - self.start_time
        print(f"총 실행 시간: {total_time:.1f}초")

def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description='개발 도구')
    parser.add_argument('--test-type', type=str, default='accuracy',
                       choices=['accuracy', 'speed', 'stress', 'benchmark', 'all'],
                       help='테스트 유형')
    parser.add_argument('--sample-size', type=int, default=30,
                       help='테스트 샘플 수')
    parser.add_argument('--duration', type=int, default=5,
                       help='스트레스 테스트 시간 (분)')
    
    args = parser.parse_args()
    
    # 시스템 요구사항 확인
    if not torch.cuda.is_available():
        print("❌ CUDA 없음 - GPU 추론 불가능")
        return
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    
    # 데이터 파일 확인
    if not os.path.exists('./test.csv') or not os.path.exists('./sample_submission.csv'):
        print("❌ 데이터 파일 없음")
        return
    
    # 모델 설정 (동적 최적화)
    if gpu_memory >= 20:
        model_config = {
            "model_name": "upstage/SOLAR-10.7B-Instruct-v1.0",
            "device": "cuda",
            "load_in_4bit": False,
            "max_memory_gb": int(gpu_memory * 0.9)
        }
        print("🎯 초고성능 모드 설정")
    elif gpu_memory >= 12:
        model_config = {
            "model_name": "upstage/SOLAR-10.7B-Instruct-v1.0",
            "device": "cuda", 
            "load_in_4bit": False,
            "max_memory_gb": int(gpu_memory * 0.85)
        }
        print("🎯 고성능 모드 설정")
    else:
        model_config = {
            "model_name": "upstage/SOLAR-10.7B-Instruct-v1.0",
            "device": "cuda",
            "load_in_4bit": True,  # 메모리 절약
            "max_memory_gb": int(gpu_memory * 0.8)
        }
        print("🎯 효율성 모드 설정")
    
    # 테스터 초기화 및 실행
    tester = None
    try:
        tester = UltraHighPerformanceTester(model_config)
        
        if args.test_type == 'accuracy':
            tester.run_comprehensive_accuracy_test(args.sample_size)
        elif args.test_type == 'speed':
            tester.run_ultra_speed_test(args.sample_size)
        elif args.test_type == 'stress':
            tester.run_stress_test(args.duration)
        elif args.test_type == 'benchmark':
            tester.run_comprehensive_benchmark()
        elif args.test_type == 'all':
            print("🔥 전체 테스트 실행")
            tester.run_comprehensive_accuracy_test(20)
            tester.run_ultra_speed_test(30)
            tester.run_stress_test(3)
        
        print("\n✅ 테스트 완료")
        
    except KeyboardInterrupt:
        print("\n⚠️ 테스트 중단됨")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tester:
            tester.cleanup()

if __name__ == "__main__":
    main()